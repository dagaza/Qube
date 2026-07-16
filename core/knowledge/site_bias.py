"""Discovery site bias defaults for built-in composer tools."""

from __future__ import annotations

# Built-in @recipe DDG scope — not used for extractor selection (supports() only).
RECIPE_DEFAULT_SITE_BIAS: tuple[str, ...] = (
    "seriouseats.com",
    "bbcgoodfood.com",
    "allrecipes.com",
    "foodnetwork.com",
    "bonappetit.com",
)
