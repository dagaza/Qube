"""Official publisher branding resolver for Hugging Face models."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from core.paths import resource_path

logger = logging.getLogger("Qube.HFBranding")

HF_API_BASE = "https://huggingface.co/api"

# Central allowlist of trusted official model publishers.
# Keys are lowercase Hub org ids and catalog ``publisher`` values (see assets/logos/).
TRUSTED_PUBLISHERS: dict[str, dict[str, str]] = {
    "google": {"name": "Google", "logo": "Google.svg"},
    "openai": {"name": "OpenAI", "logo": "openai.svg"},
    "microsoft": {"name": "Microsoft", "logo": "microsoft.svg"},
    "meta-llama": {"name": "Meta", "logo": "meta.svg"},
    "mistralai": {"name": "Mistral", "logo": "mistral.svg"},
    "anthropic": {"name": "Anthropic", "logo": "anthropic.svg"},
    "ai2": {"name": "AI2", "logo": "Ai2.svg"},
    "deepseek-ai": {"name": "DeepSeek", "logo": "deepseek.svg"},
    "xai": {"name": "xAI", "logo": "grok.svg"},
    "grok": {"name": "xAI", "logo": "grok.svg"},
    "ibm": {"name": "IBM", "logo": "ibm.svg"},
    "ibm-granite": {"name": "IBM", "logo": "ibm.svg"},
    "moonshotai": {"name": "Kimi", "logo": "kimi.svg"},
    "kimi": {"name": "Kimi", "logo": "kimi.svg"},
    "qwen": {"name": "Qwen", "logo": "qwen.png"},
    "nvidia": {"name": "NVIDIA", "logo": "nvidia.svg"},
    "runwayml": {"name": "Runway", "logo": "runway.svg"},
    "runway": {"name": "Runway", "logo": "runway.svg"},
    "zai-org": {"name": "Z.ai", "logo": "zai.svg"},
    "rnj": {"name": "RNJ", "logo": "rnj.svg"},
}


def _logo_exists(filename: str) -> bool:
    return resource_path("assets", "logos", filename).is_file()


def _owner_from_repo_id(repo_id: str) -> str:
    rid = str(repo_id or "").strip()
    if "/" not in rid:
        return ""
    return rid.split("/", 1)[0].strip().lower()


def owner_from_repo_id(repo_id: str) -> str:
    """Public alias for repo namespace owner (segment before ``/``)."""
    return _owner_from_repo_id(repo_id)


def _humanize_owner(owner: str) -> str:
    o = str(owner or "").strip().lower()
    if not o:
        return ""
    known = {
        "unsloth": "Unsloth",
        "bartowski": "Bartowski",
        "davidau": "DavidAU",
        "mradermacher": "Mradermacher",
        "lmstudio-community": "LM Studio Community",
    }
    if o in known:
        return known[o]
    parts = [p for p in re.split(r"[-_]+", o) if p]
    if not parts:
        return o
    return " ".join(p[:1].upper() + p[1:] if len(p) > 1 else p.upper() for p in parts)


def _avatar_url_from_payload(data: dict[str, Any] | None) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("avatarUrl", "avatar_url", "avatar", "picture"):
        v = data.get(key)
        if isinstance(v, str) and v.strip().lower().startswith("http"):
            return v.strip()
    return ""


def _hf_logo_asset_path() -> str:
    for rel in ("assets/logos/hf-logo.svg", "assets/icons/hf-logo.svg"):
        if resource_path(*rel.split("/")).is_file():
            return f"/{rel}"
    return "/assets/logos/hf-logo.svg"


def _norm_card_data(model: dict[str, Any]) -> dict[str, Any]:
    card = model.get("cardData")
    if isinstance(card, dict):
        return card
    card = model.get("card_data")
    return card if isinstance(card, dict) else {}


def is_derivative_model(model: dict[str, Any]) -> bool:
    """Return True for quantized/derived variants that should not be branded as official."""
    rid = str(model.get("id", "")).strip()
    owner = _owner_from_repo_id(rid)

    card_data = _norm_card_data(model)
    base_model = card_data.get("base_model")
    if base_model:
        # Reject only when declared base model belongs to a *different* owner.
        refs: list[str] = []
        if isinstance(base_model, str) and base_model.strip():
            refs = [base_model.strip()]
        elif isinstance(base_model, list):
            refs = [str(x).strip() for x in base_model if str(x).strip()]
        for ref in refs:
            base_owner = _owner_from_repo_id(ref)
            if base_owner and owner and base_owner != owner:
                return True
    # Do not blanket-reject by "gguf/quant" keyword for official publisher namespaces.
    return False


def _is_org_verified(org_data: dict[str, Any] | None) -> bool:
    if not isinstance(org_data, dict):
        return False
    # HF responses can vary by endpoint/version; accept common flags.
    if bool(org_data.get("verified", False)) or bool(org_data.get("isVerified", False)):
        return True
    if bool(org_data.get("is_verified", False)):
        return True
    author = org_data.get("author")
    if isinstance(author, dict):
        if bool(author.get("isVerified", False)) or bool(author.get("verified", False)):
            return True
    return False


def _is_model_owner_verified(model_data: dict[str, Any] | None) -> bool:
    if not isinstance(model_data, dict):
        return False
    # Common HF model payload shape.
    author_data = model_data.get("authorData")
    if isinstance(author_data, dict):
        if bool(author_data.get("isVerified", False)) or bool(author_data.get("verified", False)):
            return True
    # Defensive fallback for alternate key casings.
    for key in ("isVerified", "verified", "is_verified"):
        if bool(model_data.get(key, False)):
            return True
    return False


def get_official_branding(model: dict[str, Any], org_data: dict[str, Any] | None) -> dict[str, Any] | None:
    owner = _owner_from_repo_id(str(model.get("id", "")))
    if owner not in TRUSTED_PUBLISHERS:
        return None
    # Prefer model payload verification signal when present; fallback to org endpoint.
    # If both are unavailable, allow trusted-owner branding rather than failing closed
    # due to transient endpoint/network issues.
    model_verified = _is_model_owner_verified(model)
    org_verified = _is_org_verified(org_data)
    if not model_verified and not org_verified and org_data is not None:
        return None

    if is_derivative_model(model):
        return None

    publisher = TRUSTED_PUBLISHERS[owner]
    logo_name = str(publisher.get("logo", "")).strip()
    if not logo_name or not _logo_exists(logo_name):
        return None

    return {
        "name": str(publisher["name"]),
        "logo": f"/assets/logos/{logo_name}",
        "official": True,
    }


class HuggingFaceBrandingResolver:
    """Caching resolver for official publisher verification + branding."""

    def __init__(self, timeout_s: float = 6.0):
        self._timeout_s = float(timeout_s)
        self._model_cache: dict[str, dict[str, Any] | None] = {}
        self._org_cache: dict[str, dict[str, Any] | None] = {}
        self._user_cache: dict[str, dict[str, Any] | None] = {}
        self._variant_cache: dict[str, dict[str, Any] | None] = {}
        self._lock = Lock()
        self._enabled = str(os.getenv("QUBE_HF_OFFICIAL_BRANDING", "1")).strip().lower() not in {"0", "false", "no"}
        self._avatar_cache_dir = Path.home() / ".qube" / "avatar_cache"

    def _fetch_json(self, url: str) -> dict[str, Any] | None:
        req = Request(url=url, headers={"User-Agent": "Qube-Desktop/1.0"})
        try:
            with urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8", errors="replace"))
            return payload if isinstance(payload, dict) else None
        except HTTPError as e:
            # 401/403/404/private/missing -> treat unverified and return no branding.
            logger.debug("HF branding fetch HTTPError for %s: %s", url, e)
            return None
        except URLError as e:
            logger.debug("HF branding fetch URLError for %s: %s", url, e)
            return None
        except Exception as e:
            logger.debug("HF branding fetch error for %s: %s", url, e)
            return None

    def get_model_metadata(self, repo_id: str) -> dict[str, Any] | None:
        rid = str(repo_id or "").strip()
        if not rid:
            return None
        with self._lock:
            if rid in self._model_cache:
                return self._model_cache[rid]
        data = self._fetch_json(f"{HF_API_BASE}/models/{rid}")
        with self._lock:
            self._model_cache[rid] = data
        return data

    def get_user_metadata(self, owner: str) -> dict[str, Any] | None:
        owner_l = str(owner or "").strip().lower()
        if not owner_l:
            return None
        with self._lock:
            if owner_l in self._user_cache:
                return self._user_cache[owner_l]
        data = self._fetch_json(f"{HF_API_BASE}/users/{owner_l}")
        with self._lock:
            self._user_cache[owner_l] = data
        return data

    def get_org_metadata(self, owner: str) -> dict[str, Any] | None:
        owner_l = str(owner or "").strip().lower()
        if not owner_l:
            return None
        with self._lock:
            if owner_l in self._org_cache:
                return self._org_cache[owner_l]
        data = self._fetch_json(f"{HF_API_BASE}/organizations/{owner_l}")
        with self._lock:
            self._org_cache[owner_l] = data
        return data

    def resolve_for_model(self, repo_id: str, preloaded_model: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Resolve branding for a model repo id, returning None when not official."""
        if not self._enabled:
            return None

        owner = _owner_from_repo_id(repo_id)
        if owner not in TRUSTED_PUBLISHERS:
            logger.debug("HF branding skipped (%s): owner not in allowlist.", repo_id)
            return None

        model_data = self.get_model_metadata(repo_id) or {}
        if not isinstance(model_data, dict):
            model_data = {}
        if preloaded_model:
            # Keep network payload authoritative, but fill missing fields from preloaded list data.
            for key in ("id", "cardData", "card_data"):
                if key not in model_data and key in preloaded_model:
                    model_data[key] = preloaded_model[key]
        if "id" not in model_data:
            model_data["id"] = str(repo_id or "")

        org_data = self.get_org_metadata(owner)
        branding = get_official_branding(model_data, org_data)
        if branding is None:
            if is_derivative_model(model_data):
                logger.debug("HF branding rejected (%s): derivative model.", repo_id)
            elif org_data is not None and not _is_org_verified(org_data) and not _is_model_owner_verified(model_data):
                logger.debug("HF branding rejected (%s): org unverified/unavailable.", repo_id)
            else:
                logger.debug("HF branding rejected (%s): missing logo or policy mismatch.", repo_id)
        return branding

    def _cache_avatar_file(self, owner: str, avatar_url: str) -> Path | None:
        owner_l = str(owner or "").strip().lower()
        url = str(avatar_url or "").strip()
        if not owner_l or not url:
            return None
        try:
            self._avatar_cache_dir.mkdir(parents=True, exist_ok=True)
            ext = ".png"
            if ".jpg" in url.lower() or ".jpeg" in url.lower():
                ext = ".jpg"
            elif ".svg" in url.lower():
                ext = ".svg"
            dest = self._avatar_cache_dir / f"{owner_l}{ext}"
            if dest.is_file() and (time.time() - dest.stat().st_mtime) < 7 * 86400:
                return dest
            req = Request(url=url, headers={"User-Agent": "Qube-Desktop/1.0"})
            with urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
            if not raw:
                return None
            dest.write_bytes(raw)
            return dest
        except Exception as e:
            logger.debug("HF avatar cache failed for %s: %s", owner_l, e)
            return None

    def resolve_variant_branding(self, repo_id: str) -> dict[str, Any] | None:
        """
        Branding for the GGUF repo publisher (community quantizer), not the official allowlist.
        Uses HF org/user API for display name + avatar when available; falls back to HF logo asset.
        """
        rid = str(repo_id or "").strip()
        if not rid:
            return None
        with self._lock:
            if rid in self._variant_cache:
                cached = self._variant_cache[rid]
                return dict(cached) if isinstance(cached, dict) else None

        owner = _owner_from_repo_id(rid)
        if not owner:
            return None

        org_data = self.get_org_metadata(owner)
        user_data = None
        if not org_data:
            user_data = self.get_user_metadata(owner)
        payload = org_data if isinstance(org_data, dict) and org_data else user_data
        payload = payload if isinstance(payload, dict) else {}

        name = ""
        for key in ("fullname", "fullName", "name", "displayName"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break
        if not name:
            name = _humanize_owner(owner)

        logo = _hf_logo_asset_path()
        avatar_url = _avatar_url_from_payload(payload)
        cached_path = self._cache_avatar_file(owner, avatar_url) if avatar_url else None
        if cached_path is not None:
            logo = str(cached_path)

        result = {
            "name": name,
            "logo": logo,
            "owner": owner,
            "official": False,
            "variant": True,
        }
        with self._lock:
            self._variant_cache[rid] = result
        return dict(result)

