"""Download files with progress indicators."""

import email.message
import logging
import mimetypes
import os
from dataclasses import dataclass
from http import HTTPStatus
from typing import BinaryIO, Iterable, Optional, Tuple

from pip._vendor.requests.models import Response
from pip._vendor.urllib3.exceptions import ReadTimeoutError

from pip._internal.cli.progress_bars import get_download_progress_renderer
from pip._internal.exceptions import IncompleteDownloadError, NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext

logger = logging.getLogger(__name__)


def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers["content-length"])
    except (ValueError, KeyError, TypeError):
        return None


def _get_http_response_etag_or_last_modified(resp: Response) -> Optional[str]:
    """
    Return either the ETag or Last-Modified header (or None if neither exists).
    The return value can be used in an If-Range header.
    """
    return resp.headers.get("etag", resp.headers.get("last-modified"))


def _log_download(
    resp: Response,
    link: Link,
    progress_bar: str,
    total_length: Optional[int],
    range_start: Optional[int] = 0,
) -> Iterable[bytes]:
    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment

    logged_url = redact_auth_from_url(url)

    if total_length:
        if range_start:
            logged_url = (
                f"{logged_url} ({format_size(range_start)}/{format_size(total_length)})"
            )
        else:
            logged_url = f"{logged_url} ({format_size(total_length)})"

    if is_from_cache(resp):
        logger.info("Using cached %s", logged_url)
    elif range_start:
        logger.info("Resuming download %s", logged_url)
    else:
        logger.info("Downloading %s", logged_url)

    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif is_from_cache(resp):
        show_progress = False
    elif not total_length:
        show_progress = True
    elif total_length > (512 * 1024):
        show_progress = True
    else:
        show_progress = False

    chunks = response_chunks(resp)

    if not show_progress:
        return chunks

    renderer = get_download_progress_renderer(
        bar_type=progress_bar, size=total_length, initial_progress=range_start
    )
    return renderer(chunks)


def sanitize_content_filename(filename: str) -> str:
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition: str, default_filename: str) -> str:
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    m = email.message.Message()
    m["content-type"] = content_disposition
    filename = m.get_param("filename")
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(str(filename))
    return filename or default_filename


def _get_http_response_filename(resp: Response, link: Link) -> str:
    """Get an ideal filename from the given HTTP response, falling back to
    the link filename if not provided.
    """
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get("content-disposition")
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext: Optional[str] = splitext(filename)[1]
    if not ext:
        ext = mimetypes.guess_extension(resp.headers.get("content-type", ""))
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    return filename


@dataclass(frozen=True)
class Downloader:
    session: PipSession
    progress_bar: str
    resume_retries: int

    def batch(
        self, links: Iterable[Link], location: str
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        """Convenience method to download multiple links."""
        for link in links:
            filepath, content_type = self(link, location)
            yield link, (filepath, content_type)

    def __call__(self, link: Link, location: str) -> Tuple[str, str]:
        """Download a link and save it under location."""
        return _FileDownloader(
            self.session, self.progress_bar, self.resume_retries
        ).download(link, location)


class _FileDownloader:
    """Single-use helper for downloading a single link."""

    # To better understand the download resumption logic, see the mdn web docs:
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests

    def __init__(
        self, session: PipSession, progress_bar: str, resume_retries: int
    ) -> None:
        assert resume_retries >= 0, "retries must be bigger or equal to zero"
        self._session = session
        self._progress_bar = progress_bar
        self._resume_retries = resume_retries

        self.link: Link
        self.output_file: BinaryIO
        self.file_length: Optional[int]
        self.bytes_received = 0
        self.resumes_left = resume_retries

    def download(self, link: Link, location: str) -> Tuple[str, str]:
        """Download the link and save it under location."""
        assert not hasattr(self, "link"), "file downloader already used"
        self.link = link
        resp = self._request_download()
        self.file_length = _get_http_response_size(resp)

        filepath = os.path.join(location, _get_http_response_filename(resp, link))
        with open(filepath, "wb") as self.output_file:
            self._process_response(resp)
            if self._is_incomplete():
                self._attempt_resumes_or_redownloads(resp)

        content_type = resp.headers.get("Content-Type", "")
        return filepath, content_type

    def _is_incomplete(self) -> bool:
        return bool(self.file_length and self.bytes_received < self.file_length)

    def _request_download(
        self, allow_range: bool = False, must_match: Optional[Response] = None
    ) -> Response:
        """Issue a GET request to start downloading chunks."""
        target_url = self.link.url_without_fragment
        headers = HEADERS.copy()
        if allow_range:
            # If the download has already started, request a partial download.
            if start_at := self.bytes_received:
                headers["Range"] = f"bytes={start_at}-"
            # If possible, use a conditional range request to avoid corrupted
            # downloads caused by the remote file changing in-between.
            if identifier := _get_http_response_etag_or_last_modified(must_match):
                headers["If-Range"] = identifier
        try:
            resp = self._session.get(target_url, headers=headers, stream=True)
            raise_for_status(resp)
        except NetworkConnectionError as e:
            assert e.response is not None
            status_code = e.response.status_code
            logger.critical("HTTP error %s while getting %s", status_code, self.link)
            raise

        return resp

    def _process_response(self, resp: Response) -> None:
        """Download and save chunks from a response."""
        chunks = _log_download(
            resp,
            self.link,
            self._progress_bar,
            self.file_length,
            range_start=self.bytes_received,
        )
        try:
            for chunk in chunks:
                self.bytes_received += len(chunk)
                self.output_file.write(chunk)
        except ReadTimeoutError as e:
            # If the file length is not known, then give up downloading the file.
            if self.file_length is None:
                raise e

            logger.warning("Connection timed out while downloading.")

    def _attempt_resumes_or_redownloads(self, resp: Response) -> None:
        """Attempt to resume/restart the download if connection was dropped."""

        while self.resumes_left and self._is_incomplete():
            assert self.file_length is not None
            logger.warning(
                "Attempting to resume incomplete download (%s/%s, attempt %d)",
                format_size(self.bytes_received),
                format_size(self.file_length),
                (self._resume_retries - self.resumes_left),
            )
            self.resumes_left -= 1

            try:
                # Try to resume the download using a HTTP range request.
                resume_resp = self._request_download(allow_range=True, must_match=resp)
                # Fallback: if the server responded with 200 (i.e., the file has
                # since been modified or range requests are unsupported) or any
                # other unexpected status, restart the download from the beginning.
                must_restart = resume_resp.status_code != HTTPStatus.PARTIAL_CONTENT
                if must_restart:
                    self.output_file.truncate(0)
                    self.bytes_received = 0
                    self.file_length = _get_http_response_size(resp)
                    resp = resume_resp

                self._process_response(resume_resp)
            except (ConnectionError, ReadTimeoutError, OSError):
                continue

        # No more resume attempts. Raise an error if the download is still incomplete.
        if self._is_incomplete():
            assert self.file_length is not None
            os.remove(self.output_file.name)
            raise IncompleteDownloadError(
                self.link,
                self.bytes_received,
                self.file_length,
                retries=self._resume_retries,
            )
