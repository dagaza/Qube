"""Classify Hugging Face Hub / HTTP failures into user-facing HubErrorInfo payloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

HF_STATUS_URL = "https://status.huggingface.co/"


class HubErrorKind(str, Enum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    SERVER = "server"
    RATE_LIMIT = "rate_limit"
    NOT_FOUND = "not_found"
    CLIENT = "client"
    DISK = "disk"
    VALIDATION = "validation"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HubErrorInfo:
    kind: HubErrorKind
    title: str
    message: str
    technical_detail: str = ""
    retryable: bool = False
    show_status_link: bool = False

    @property
    def is_platform_outage(self) -> bool:
        return self.kind in {
            HubErrorKind.CONNECTION,
            HubErrorKind.TIMEOUT,
            HubErrorKind.SERVER,
            HubErrorKind.RATE_LIMIT,
        }

    @property
    def inline_only(self) -> bool:
        """Prefer inline hub hint over a modal (search network blips)."""
        return self.is_platform_outage

    def dialog_message(self) -> str:
        if self.technical_detail and self.technical_detail not in self.message:
            return f"{self.message}\n\n({self.technical_detail})"
        return self.message


def _detail_from_exc(exc: BaseException | str) -> str:
    if isinstance(exc, BaseException):
        return str(exc).strip()
    return str(exc or "").strip()


def _message_suggests_connection(text: str) -> bool:
    low = text.lower()
    needles = (
        "connection refused",
        "connection reset",
        "connection aborted",
        "failed to establish a new connection",
        "name or service not known",
        "nodename nor servname",
        "network is unreachable",
        "temporary failure in name resolution",
        "getaddrinfo failed",
        "ssl:",
        "certificate",
        "remote disconnected",
    )
    return any(n in low for n in needles)


def _message_suggests_timeout(text: str) -> bool:
    low = text.lower()
    return "timed out" in low or "timeout" in low


def classify_hf_http_status(status_code: int, *, context: str = "") -> HubErrorInfo:
    code = int(status_code)
    ctx = f" {context}".rstrip()
    if code == 404:
        return HubErrorInfo(
            kind=HubErrorKind.NOT_FOUND,
            title="Not found on Hugging Face",
            message=(
                "That repository or file was not found on the Hub."
                " It may have moved, been renamed, or is private."
            ),
            technical_detail=f"HTTP {code}{ctx}",
            retryable=False,
            show_status_link=False,
        )
    if code == 429:
        return HubErrorInfo(
            kind=HubErrorKind.RATE_LIMIT,
            title="Hugging Face rate limit",
            message=(
                "Hugging Face is limiting requests right now. "
                "Wait a minute and try again."
            ),
            technical_detail=f"HTTP {code}{ctx}",
            retryable=True,
            show_status_link=True,
        )
    if 500 <= code <= 599:
        return HubErrorInfo(
            kind=HubErrorKind.SERVER,
            title="Hugging Face unavailable",
            message=(
                "Hugging Face returned a server error. "
                "This is usually temporary — try again in a few minutes."
            ),
            technical_detail=f"HTTP {code}{ctx}",
            retryable=True,
            show_status_link=True,
        )
    if 400 <= code <= 499:
        return HubErrorInfo(
            kind=HubErrorKind.CLIENT,
            title="Request rejected",
            message=(
                "Hugging Face rejected this request. "
                "Check the repository id and file name, then try again."
            ),
            technical_detail=f"HTTP {code}{ctx}",
            retryable=False,
            show_status_link=False,
        )
    return HubErrorInfo(
        kind=HubErrorKind.UNKNOWN,
        title="Hugging Face error",
        message="Something went wrong while talking to Hugging Face.",
        technical_detail=f"HTTP {code}{ctx}",
        retryable=True,
        show_status_link=True,
    )


def classify_hf_error(
    exc: BaseException | str,
    *,
    http_status: int | None = None,
    context: str = "",
) -> HubErrorInfo:
    """Map exceptions / HTTP codes / message strings to a HubErrorInfo."""
    if http_status is not None:
        return classify_hf_http_status(http_status, context=context)

    detail = _detail_from_exc(exc)

    if isinstance(exc, OSError) and not isinstance(exc, ConnectionError):
        low = detail.lower()
        if "no space left" in low or "disk" in low or "enospc" in low:
            return HubErrorInfo(
                kind=HubErrorKind.DISK,
                title="Not enough disk space",
                message="There is not enough free disk space to finish this operation.",
                technical_detail=detail,
                retryable=False,
                show_status_link=False,
            )

    if isinstance(exc, ValueError) or (
        detail and re.match(r"^(invalid|empty|file must)", detail.lower())
    ):
        return HubErrorInfo(
            kind=HubErrorKind.VALIDATION,
            title="Invalid request",
            message=detail or "The request was not valid.",
            technical_detail=detail,
            retryable=False,
            show_status_link=False,
        )

    # requests / urllib3 exception types without importing requests at module load
    exc_name = type(exc).__name__ if isinstance(exc, BaseException) else ""
    if exc_name in {"ConnectionError", "ConnectTimeout", "ProxyError", "SSLError"}:
        return HubErrorInfo(
            kind=HubErrorKind.CONNECTION,
            title="Can't reach Hugging Face",
            message=(
                "Qube could not connect to Hugging Face. "
                "Check your internet connection, then try again."
            ),
            technical_detail=detail,
            retryable=True,
            show_status_link=True,
        )
    if exc_name in {"ReadTimeout", "Timeout", "TimeoutError"}:
        return HubErrorInfo(
            kind=HubErrorKind.TIMEOUT,
            title="Hugging Face timed out",
            message=(
                "The request to Hugging Face took too long and was cancelled. "
                "Try again when your connection is stable or HF is back online."
            ),
            technical_detail=detail,
            retryable=True,
            show_status_link=True,
        )

    if isinstance(exc, ConnectionError) or _message_suggests_connection(detail):
        return HubErrorInfo(
            kind=HubErrorKind.CONNECTION,
            title="Can't reach Hugging Face",
            message=(
                "Qube could not connect to Hugging Face. "
                "Check your internet connection, then try again."
            ),
            technical_detail=detail,
            retryable=True,
            show_status_link=True,
        )

    if isinstance(exc, TimeoutError) or _message_suggests_timeout(detail):
        return HubErrorInfo(
            kind=HubErrorKind.TIMEOUT,
            title="Hugging Face timed out",
            message=(
                "The request to Hugging Face took too long and was cancelled. "
                "Try again when your connection is stable or HF is back online."
            ),
            technical_detail=detail,
            retryable=True,
            show_status_link=True,
        )

    # Parse embedded HTTP codes from worker messages
    m = re.search(r"\bHTTP\s+(\d{3})\b", detail, re.I)
    if m:
        return classify_hf_http_status(int(m.group(1)), context=context)

    if detail.lower().startswith("http "):
        try:
            code = int(detail.split()[1])
            return classify_hf_http_status(code, context=context)
        except (IndexError, ValueError):
            pass

    return HubErrorInfo(
        kind=HubErrorKind.UNKNOWN,
        title="Hugging Face error",
        message=(
            "Something went wrong while talking to Hugging Face. "
            "Try again, or browse Qube Verified models while offline."
        ),
        technical_detail=detail,
        retryable=True,
        show_status_link=True,
    )


def coerce_hub_error(value: Any) -> HubErrorInfo:
    """Accept HubErrorInfo or legacy plain-string errors from workers."""
    if isinstance(value, HubErrorInfo):
        return value
    text = str(value or "").strip()
    if not text:
        return HubErrorInfo(
            kind=HubErrorKind.UNKNOWN,
            title="Hugging Face error",
            message="Something went wrong while talking to Hugging Face.",
            retryable=True,
            show_status_link=True,
        )
    return classify_hf_error(text)
