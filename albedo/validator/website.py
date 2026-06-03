"""albedo.validator.website — Upload the dashboard website to Hippius S3.

HTML files live in website/html/ but are served from the S3 bucket root, so
relative asset paths written for the local html/ subdirectory (``"../css/"``)
are rewritten to ``"./css/"`` before upload.

JS modules, CSS, favicons, and llms.txt are uploaded at their paths relative
to the website/ root so ES module imports resolve correctly in the browser.

dashboard.json is intentionally excluded — it is managed by State.flush_dashboard().
"""
from __future__ import annotations

import hashlib
import logging
import os

from albedo.storage.store import ObjectStore

log = logging.getLogger(__name__)

# Path to website/ relative to this module (albedo/validator/ → ../../website)
_HERE        = os.path.dirname(os.path.abspath(__file__))
_WEBSITE_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "website"))

_CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
    ".png":  "image/png",
    ".txt":  "text/plain; charset=utf-8",
}

# Extensions we never upload as static assets.
_SKIP_EXTENSIONS = {".json", ".bak", ".md", ".pyc", ".py"}


def upload_website(store: ObjectStore, website_dir: str = _WEBSITE_DIR) -> str | None:
    """Upload all website assets to the Hippius dashboard bucket.

    Returns the build_id hex string, or None if website/html/ was not found.
    """
    html_dir = os.path.join(website_dir, "html")
    if not os.path.isdir(html_dir):
        log.warning("website/html/ not found at %s — skipping website upload", html_dir)
        return None

    html_files = sorted(f for f in os.listdir(html_dir) if f.endswith(".html"))
    if not html_files:
        log.warning("no .html files found in %s", html_dir)
        return None

    # Build ID: hash of all HTML source combined, truncated to 12 hex chars.
    combined = b"".join(
        open(os.path.join(html_dir, f), "rb").read() for f in html_files
    )
    build_id = hashlib.sha256(combined).hexdigest()[:12]

    # --- HTML files --------------------------------------------------------
    # Served from the S3 root (e.g. /albedo/index.html), not from html/.
    # Strip the leading "../" from all asset hrefs/srcs so paths resolve from root.
    for fname in html_files:
        with open(os.path.join(html_dir, fname), "rb") as fh:
            data = fh.read()
        data = data.replace(b"__BUILD_ID__", build_id.encode())
        data = data.replace(b'"../', b'"./')   # href="../css/" → href="./css/"
        data = data.replace(b"'../", b"'./")   # src='../js/' → src='./js/'
        ok = store.put_dashboard_raw(
            fname, data, "text/html; charset=utf-8",
            cache_control="no-cache, must-revalidate",
        )
        log.debug("uploaded %s (ok=%s)", fname, ok)

    # --- Static assets (CSS, JS modules, favicons, llms.txt, …) -----------
    # Walk website/ tree, skip html/ (already handled) and data files.
    for root, dirs, files in os.walk(website_dir):
        dirs[:] = sorted(d for d in dirs if d != "html" and not d.startswith("."))
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in _SKIP_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            # S3 key preserves the relative path from website/ root.
            s3_key = os.path.relpath(fpath, website_dir).replace(os.sep, "/")
            content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")
            with open(fpath, "rb") as fh:
                data = fh.read()
            ok = store.put_dashboard_raw(
                s3_key, data, content_type,
                cache_control="no-cache, must-revalidate",
            )
            log.debug("uploaded %s (ok=%s)", s3_key, ok)

    return build_id


def log_hippius_urls(build_id: str, bucket: str) -> None:
    """Log the public Hippius URLs for the uploaded website."""
    base = f"https://us-east-1.hippius.com/{bucket}"
    log.info("website live at %s/index.html (build=%s)", base, build_id)
    log.info("  kings page : %s/kings.html", base)
    log.info("  dashboard  : %s/dashboard.json", base)
    log.info("  llms.txt   : %s/llms.txt", base)
