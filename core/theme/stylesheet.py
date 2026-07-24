"""Stylesheet rendering from resolved themes."""

from __future__ import annotations

import re
from dataclasses import fields
from functools import lru_cache

from core.paths import resource_path
from core.theme.color_utils import parse_color
from core.theme.resolver import ThemeResolver
from core.theme.schemes import BUILTIN_SCHEMES, default_scheme_id_for_mode
from core.theme.tokens import ResolvedTheme, ThemeMode

_COLOR_LITERAL_RE = re.compile(
    r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})"
    r"|rgba\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[0-9.]+\s*\)",
)
_TOKEN_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def _normalize_color_literal(value: str) -> str:
    return parse_color(value).to_rgba().replace(" ", "")


def _is_color_field_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        _normalize_color_literal(value)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=2)
def _reference_theme(mode: ThemeMode) -> ResolvedTheme:
    scheme_id = default_scheme_id_for_mode(mode.value)
    return ThemeResolver(BUILTIN_SCHEMES).resolve(mode=mode, scheme_id=scheme_id)


@lru_cache(maxsize=2)
def _literal_to_token_name(mode: ThemeMode) -> dict[str, str]:
    """Map normalized reference literals → ``ResolvedTheme`` field names.

    When several tokens share the same reference color, the earliest field in
    ``ResolvedTheme`` wins (e.g. ``accent`` over ``chat_header``) so primitive
    overrides propagate into template literals.
    """
    reference = _reference_theme(mode)
    mapping: dict[str, str] = {}
    for field in fields(reference):
        value = getattr(reference, field.name)
        if not _is_color_field_value(value):
            continue
        normalized = _normalize_color_literal(value)
        stripped = value.strip()
        if normalized not in mapping:
            mapping[normalized] = field.name
        if stripped not in mapping:
            mapping[stripped] = field.name
    return mapping


def _static_template_path(mode: ThemeMode):
    filename = "base.qss" if mode.is_dark else "light.qss"
    return resource_path("assets", "styles", filename)


def render_stylesheet(resolved: ResolvedTheme) -> str:
    """Render ephemeral QSS by substituting template literals with token values.

    Templates remain ``assets/styles/base.qss`` and ``light.qss`` structurally.
    Reference literals from the built-in scheme for the active mode are mapped to
    token names, then filled from ``resolved``. Unmapped literals are preserved.
    When a token value is unchanged from the reference scheme, the original literal
    formatting in the template is kept (e.g. rgba spacing).
    """
    path = _static_template_path(resolved.mode)
    template = path.read_text(encoding="utf-8")

    def _replace_placeholder(match: re.Match[str]) -> str:
        token_name = match.group(1)
        try:
            return str(getattr(resolved, token_name))
        except AttributeError:
            return match.group(0)

    template = _TOKEN_PLACEHOLDER_RE.sub(_replace_placeholder, template)
    token_by_literal = _literal_to_token_name(resolved.mode)
    reference = _reference_theme(resolved.mode)

    def _replace_literal(match: re.Match[str]) -> str:
        literal = match.group(0)
        try:
            normalized = _normalize_color_literal(literal)
        except ValueError:
            return literal
        token_name = token_by_literal.get(literal) or token_by_literal.get(normalized)
        if not token_name:
            return literal
        replacement = getattr(resolved, token_name)
        reference_value = getattr(reference, token_name)
        try:
            if _normalize_color_literal(replacement) == _normalize_color_literal(
                reference_value
            ):
                return literal
        except ValueError:
            pass
        return replacement

    return _COLOR_LITERAL_RE.sub(_replace_literal, template)
