"""Tests for GitHub repository URL validation and loading."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock

import pytest
import requests

from src.github_loader import GitHubLoader, GitHubLoaderError, parse_github_url


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


class TestParseGitHubUrl:
    def test_valid_urls(self) -> None:
        ref1 = parse_github_url("https://github.com/octocat/Hello-World")
        assert ref1.owner == "octocat"
        assert ref1.repo == "Hello-World"
        assert ref1.branch is None
        assert ref1.canonical_url == "https://github.com/octocat/Hello-World"

        ref2 = parse_github_url("https://github.com/octocat/Hello-World.git/")
        assert ref2.owner == "octocat"
        assert ref2.repo == "Hello-World"

        ref3 = parse_github_url("https://github.com/octocat/Hello-World/tree/feature/rag")
        assert ref3.branch == "feature/rag"

    @pytest.mark.parametrize(
        "bad_url",
        [
            "",
            "   ",
            "github.com/owner/repo",
            "https://gitlab.com/owner/repo",
            "https://github.com/only-owner",
            "https://github.com/../repo",
            "not-a-url",
        ],
    )
    def test_invalid_urls_raise(self, bad_url: str) -> None:
        with pytest.raises(GitHubLoaderError):
            parse_github_url(bad_url)


class TestGitHubLoader:
    def test_load_public_repo_mocked(self) -> None:
        zip_bytes = _make_zip_bytes(
            {
                "octocat-hello-abc123/README.md": b"# Hello World\nSample repo.",
                "octocat-hello-abc123/src/app.py": b"def main():\n    return 42\n",
            }
        )

        meta_resp = MagicMock()
        meta_resp.status_code = 200
        meta_resp.headers = {}
        meta_resp.json.return_value = {
            "default_branch": "main",
            "size": 120,
            "description": "Sample repository",
            "language": "Python",
            "stargazers_count": 10,
            "forks_count": 2,
            "html_url": "https://github.com/octocat/hello",
        }

        zip_resp = MagicMock()
        zip_resp.status_code = 200
        zip_resp.headers = {}
        zip_resp.iter_content.return_value = [zip_bytes]

        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [meta_resp, zip_resp]

        loader = GitHubLoader(session=session)
        bundle = loader.load("https://github.com/octocat/hello")

        assert bundle.project_name == "hello"
        assert bundle.input_source == "github_url"
        assert bundle.metadata["branch"] == "main"
        paths = sorted(f.relative_path for f in bundle.raw_files)
        assert paths == ["README.md", "src/app.py"]

    def test_missing_repository_404(self) -> None:
        resp = MagicMock()
        resp.status_code = 404
        resp.headers = {}
        resp.text = "Not Found"

        session = MagicMock(spec=requests.Session)
        session.get.return_value = resp

        loader = GitHubLoader(session=session)
        with pytest.raises(GitHubLoaderError, match="not found"):
            loader.load("https://github.com/octocat/does-not-exist")

    def test_rate_limit_403(self) -> None:
        resp = MagicMock()
        resp.status_code = 403
        resp.headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"}
        resp.text = "API rate limit exceeded"

        session = MagicMock(spec=requests.Session)
        session.get.return_value = resp

        loader = GitHubLoader(session=session)
        with pytest.raises(GitHubLoaderError, match="rate limit"):
            loader.load("https://github.com/octocat/hello")

    def test_empty_repository_size_zero(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = {"default_branch": "main", "size": 0}

        session = MagicMock(spec=requests.Session)
        session.get.return_value = resp

        loader = GitHubLoader(session=session)
        with pytest.raises(GitHubLoaderError, match="empty"):
            loader.load("https://github.com/octocat/empty-repo")

    def test_network_timeout(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.Timeout("timed out")

        loader = GitHubLoader(session=session)
        with pytest.raises(GitHubLoaderError, match="Timed out"):
            loader.load("https://github.com/octocat/hello")
