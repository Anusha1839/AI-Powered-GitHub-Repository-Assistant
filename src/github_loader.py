"""GitHub repository loader using the GitHub REST API and safe archive extraction."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import requests

import config
from src.repository_processor import ProjectDocumentBundle, RawProjectFile
from utils.helpers import normalize_rel_path
from utils.logger import get_logger

logger = get_logger(__name__)

_GITHUB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class GitHubLoaderError(Exception):
    """Raised when a GitHub repository cannot be validated or fetched."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class GitHubRepoRef:
    """Parsed reference to a public GitHub repository."""

    owner: str
    repo: str
    branch: Optional[str] = None
    canonical_url: str = ""


def parse_github_url(url: str) -> GitHubRepoRef:
    """Validate and extract owner, repository name, and optional branch from a GitHub URL."""
    if not url or not isinstance(url, str):
        raise GitHubLoaderError("Please enter a valid GitHub repository URL.")

    raw = url.strip()
    if not raw.startswith(("http://", "https://")):
        raise GitHubLoaderError(
            "Invalid GitHub URL. Please include https:// (e.g., https://github.com/owner/repo)."
        )

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if host not in ("github.com", "www.github.com"):
        raise GitHubLoaderError(
            "URL must point to github.com (e.g., https://github.com/owner/repo)."
        )

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise GitHubLoaderError(
            "Incomplete GitHub URL. Expected format: https://github.com/owner/repository-name"
        )

    owner = path_parts[0]
    repo = path_parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    if (
        not owner
        or not repo
        or owner in (".", "..")
        or repo in (".", "..")
        or not _GITHUB_NAME_RE.match(owner)
        or not _GITHUB_NAME_RE.match(repo)
    ):
        raise GitHubLoaderError(
            "Invalid GitHub owner or repository name in URL."
        )

    branch: Optional[str] = None
    if len(path_parts) >= 4 and path_parts[2] == "tree":
        branch = "/".join(path_parts[3:])

    canonical_url = f"https://github.com/{owner}/{repo}"
    return GitHubRepoRef(
        owner=owner,
        repo=repo,
        branch=branch,
        canonical_url=canonical_url,
    )


class GitHubLoader:
    """Fetches public GitHub repository metadata and source files."""

    API_BASE = "https://api.github.com"

    def __init__(
        self,
        timeout: int = config.GITHUB_REQUEST_TIMEOUT,
        max_archive_bytes: int = config.MAX_UPLOAD_SIZE_BYTES,
        max_total_extracted_bytes: int = config.MAX_TOTAL_EXTRACTED_BYTES,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.max_archive_bytes = max_archive_bytes
        self.max_total_extracted_bytes = max_total_extracted_bytes
        self.session = session or requests.Session()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "AI-Powered-GitHub-Repository-Assistant",
        }
        token = config.get_github_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _handle_error_response(
        self, response: requests.Response, repo_ref: GitHubRepoRef
    ) -> None:
        status = response.status_code
        remaining = response.headers.get("X-RateLimit-Remaining", "")
        reset_ts = response.headers.get("X-RateLimit-Reset", "")

        if status == 404:
            raise GitHubLoaderError(
                f"Repository '{repo_ref.owner}/{repo_ref.repo}' was not found or is private.",
                status_code=404,
            )
        if status == 401:
            raise GitHubLoaderError(
                "GitHub authentication failed. Please check your optional GITHUB_TOKEN.",
                status_code=401,
            )
        if status in (403, 429):
            if remaining == "0" or "rate limit" in response.text.lower():
                hint = (
                    f" (Rate limit reset epoch: {reset_ts})" if reset_ts else ""
                )
                raise GitHubLoaderError(
                    "GitHub API rate limit exceeded. Wait a few minutes or configure "
                    f"GITHUB_TOKEN in your .env file.{hint}",
                    status_code=status,
                )
            raise GitHubLoaderError(
                f"Access to repository '{repo_ref.owner}/{repo_ref.repo}' was forbidden (HTTP {status}).",
                status_code=status,
            )
        if status == 409:
            raise GitHubLoaderError(
                f"Repository '{repo_ref.owner}/{repo_ref.repo}' is empty.",
                status_code=409,
            )
        if status >= 400:
            raise GitHubLoaderError(
                f"GitHub API returned an error (HTTP {status}) for '{repo_ref.owner}/{repo_ref.repo}'.",
                status_code=status,
            )

    def fetch_metadata(self, repo_ref: GitHubRepoRef) -> dict[str, Any]:
        """Fetch repository metadata from the GitHub REST API."""
        url = f"{self.API_BASE}/repos/{repo_ref.owner}/{repo_ref.repo}"
        try:
            response = self.session.get(
                url, headers=self._build_headers(), timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise GitHubLoaderError(
                "Timed out while connecting to GitHub. Please check your network and try again."
            ) from exc
        except requests.RequestException as exc:
            raise GitHubLoaderError(
                f"Network error while connecting to GitHub: {exc}"
            ) from exc

        self._handle_error_response(response, repo_ref)

        try:
            data = response.json()
        except ValueError as exc:
            raise GitHubLoaderError("Received an invalid JSON response from GitHub.") from exc

        if not isinstance(data, dict):
            raise GitHubLoaderError("Unexpected metadata format returned by GitHub.")

        if data.get("size", 0) == 0:
            raise GitHubLoaderError(
                f"Repository '{repo_ref.owner}/{repo_ref.repo}' is empty."
            )

        return data

    def _download_zipball(
        self, repo_ref: GitHubRepoRef, branch: str
    ) -> bytes:
        """Download repository archive ZIP in a single API call."""
        url = f"{self.API_BASE}/repos/{repo_ref.owner}/{repo_ref.repo}/zipball/{branch}"
        try:
            response = self.session.get(
                url,
                headers=self._build_headers(),
                timeout=self.timeout,
                stream=True,
            )
        except requests.Timeout as exc:
            raise GitHubLoaderError(
                "Timed out while downloading repository archive from GitHub."
            ) from exc
        except requests.RequestException as exc:
            raise GitHubLoaderError(
                f"Network error while downloading repository archive: {exc}"
            ) from exc

        self._handle_error_response(response, repo_ref)

        chunks: list[bytes] = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > self.max_archive_bytes:
                raise GitHubLoaderError(
                    f"Repository archive exceeds the maximum allowed size of "
                    f"{self.max_archive_bytes // (1024 * 1024)} MB."
                )
            chunks.append(chunk)

        archive_bytes = b"".join(chunks)
        if not archive_bytes:
            raise GitHubLoaderError(
                f"Downloaded archive for '{repo_ref.owner}/{repo_ref.repo}' is empty."
            )
        return archive_bytes

    def _extract_zipball_files(self, archive_bytes: bytes) -> list[RawProjectFile]:
        """Extract files safely from a GitHub zipball while stripping the commit root folder."""
        try:
            zf = zipfile.ZipFile(io.BytesIO(archive_bytes))
        except zipfile.BadZipFile as exc:
            raise GitHubLoaderError(
                "GitHub returned a corrupted or invalid repository archive."
            ) from exc

        raw_files: list[RawProjectFile] = []
        total_uncompressed = 0

        with zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if not infos:
                raise GitHubLoaderError("The GitHub repository contains no files.")

            # GitHub zipballs wrap everything in "<owner>-<repo>-<sha>/"
            first_segments = {
                info.filename.replace("\\", "/").split("/", 1)[0]
                for info in infos
                if "/" in info.filename.replace("\\", "/")
            }
            strip_prefix = len(first_segments) == 1

            for info in infos:
                # Skip symlinks inside archives
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    continue

                total_uncompressed += max(0, info.file_size)
                if total_uncompressed > self.max_total_extracted_bytes:
                    raise GitHubLoaderError(
                        "Extracted repository size exceeds the maximum safety limit."
                    )

                raw_name = info.filename.replace("\\", "/")
                if strip_prefix and "/" in raw_name:
                    raw_name = raw_name.split("/", 1)[1]

                rel_path = normalize_rel_path(raw_name)
                if not rel_path or ".." in rel_path.split("/"):
                    continue

                with zf.open(info) as member:
                    content = member.read()

                raw_files.append(
                    RawProjectFile(
                        relative_path=rel_path,
                        content=content,
                        size_bytes=len(content),
                    )
                )

        if not raw_files:
            raise GitHubLoaderError("No files could be extracted from the repository.")

        return raw_files

    def load(self, repo_url: str) -> ProjectDocumentBundle:
        """Load a public GitHub repository into the common ProjectDocumentBundle format."""
        repo_ref = parse_github_url(repo_url)
        logger.info("Fetching GitHub repository: %s", repo_ref.canonical_url)

        repo_meta = self.fetch_metadata(repo_ref)
        branch = repo_ref.branch or repo_meta.get("default_branch") or "main"
        archive_bytes = self._download_zipball(repo_ref, branch)
        raw_files = self._extract_zipball_files(archive_bytes)

        metadata = {
            "owner": repo_ref.owner,
            "repo": repo_ref.repo,
            "branch": branch,
            "description": repo_meta.get("description") or "",
            "primary_language": repo_meta.get("language") or "Unknown",
            "stars": repo_meta.get("stargazers_count", 0),
            "forks": repo_meta.get("forks_count", 0),
            "html_url": repo_meta.get("html_url", repo_ref.canonical_url),
        }

        return ProjectDocumentBundle(
            project_name=repo_ref.repo,
            input_source="github_url",
            source_detail=repo_ref.canonical_url,
            raw_files=raw_files,
            metadata=metadata,
        )
