"""
api/startup.py
==============
Downloads the latest artifacts.zip from GitHub Releases and unpacks it
into the project root before FastAPI boots.

Called from api/main.py lifespan, or directly:
    python -m api.startup

Artifacts zip structure (must match upload in workflow):
    artifacts/
        fifa.db
        model.pkl
        poisson_params.pkl

Environment variables:
    GITHUB_REPO         e.g. "mohdfahad20/FIFA-2026-Prediction"
    ARTIFACTS_TAG       GitHub release tag (default: "latest-data")
    ARTIFACT_NAME       zip filename (default: "artifacts.zip")

All three have sensible defaults so local dev works without any env vars
(just uses whatever .pkl / .db files are already present).
"""

import io
import logging
import os
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GITHUB_REPO = os.environ.get("GITHUB_REPO", "mohdfahad20/FIFA-2026-Prediction")
ARTIFACTS_TAG = os.environ.get("ARTIFACTS_TAG", "latest-data")
ARTIFACT_NAME = os.environ.get("ARTIFACT_NAME", "artifacts.zip")

GITHUB_API = "https://api.github.com"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files we expect inside the zip (relative to its inner folder)
EXPECTED_FILES = ["fifa.db", "model/model.pkl", "score_model/poisson_params.pkl"]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _release_download_url() -> str | None:
    """
    Resolve the download URL for ARTIFACT_NAME inside the ARTIFACTS_TAG release.
    Returns None if the release or asset is not found.
    """
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/releases/tags/{ARTIFACTS_TAG}"
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=20)
    except requests.RequestException as exc:
        log.warning("GitHub API request failed: %s", exc)
        return None

    if resp.status_code == 404:
        log.info("Release tag '%s' not found — skipping download.", ARTIFACTS_TAG)
        return None

    resp.raise_for_status()
    release = resp.json()

    for asset in release.get("assets", []):
        if asset["name"] == ARTIFACT_NAME:
            return asset["browser_download_url"]

    log.warning("Asset '%s' not found in release '%s'.", ARTIFACT_NAME, ARTIFACTS_TAG)
    return None


def _already_current(download_url: str) -> bool:
    """
    Simple freshness check: compare the ETag / Last-Modified of the remote asset
    against a cached value stored in .artifact_etag.
    Returns True when local files are already up to date.
    """
    etag_file = PROJECT_ROOT / ".artifact_etag"
    try:
        head = requests.head(download_url, timeout=10, allow_redirects=True)
        remote_etag = head.headers.get("ETag", "")
        if etag_file.exists() and etag_file.read_text().strip() == remote_etag:
            return True
        # Store for next run
        if remote_etag:
            etag_file.write_text(remote_etag)
    except Exception:
        pass  # Not fatal — just re-download
    return False


def download_and_unpack(force: bool = False) -> bool:
    """
    Download and unpack the artifacts zip.

    Returns True if files were updated, False if skipped (already current).
    Raises on download / unpack failure.
    """
    url = _release_download_url()
    if url is None:
        log.info("No remote artifacts — using local files.")
        return False

    if not force and _already_current(url):
        log.info("Local artifacts are up to date — skipping download.")
        return False

    log.info("Downloading %s …", url)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    content = b"".join(resp.iter_content(chunk_size=1 << 20))
    log.info("Downloaded %.1f MB", len(content) / 1e6)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        log.debug("Zip contents: %s", names)

        for expected in EXPECTED_FILES:
            # Support both flat zip and zip with a single top-level folder
            candidates = [n for n in names if n.endswith(f"/{expected}") or n == expected]
            if not candidates:
                log.warning("Expected file '%s' not found in zip.", expected)
                continue

            member = candidates[0]
            dest = PROJECT_ROOT / expected
            log.info("Extracting %s → %s", member, dest)

            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    log.info("Artifacts unpacked successfully.")
    return True


# ---------------------------------------------------------------------------
# Lifespan hook (called from api/main.py)
# ---------------------------------------------------------------------------

def on_startup() -> None:
    """
    Called during FastAPI lifespan startup.
    Downloads artifacts if running on Render (RENDER env var is set).
    Silently skips on local dev.
    """
    is_render = bool(os.environ.get("RENDER"))

    if not is_render:
        log.debug("Local environment — skipping artifact download.")
        return

    log.info("Render environment detected — checking for latest artifacts …")
    try:
        updated = download_and_unpack()
        if updated:
            log.info("Artifacts refreshed from GitHub Release.")
        else:
            log.info("Using existing artifacts.")
    except Exception as exc:
        # Don't crash the API if download fails — existing files may still work
        log.error("Artifact download failed: %s — continuing with existing files.", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="Download latest FIFA artifacts.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if ETag matches.")
    args = parser.parse_args()

    download_and_unpack(force=args.force)
