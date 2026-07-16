"""JSON-LD / recipe-scrapers extractor for recipe pages."""

from __future__ import annotations

import html as html_module
import json
import re
from typing import Any

from core.knowledge.document.types import Document, DocumentMetadata, DocumentSection
from core.knowledge.extractors.base import ExtractorMetadata

try:
    from recipe_scrapers import scrape_html
except ImportError:  # pragma: no cover
    scrape_html = None

EXTRACTOR_NAME = "RecipeExtractor"
EXTRACTOR_VERSION = "1.0.0"
RECIPE_CONFIDENCE = 0.98

_JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
_RECIPE_TYPE_RE = re.compile(r"\"@type\"\s*:\s*\"([^\"]+)\"", re.IGNORECASE)


def _normalize_type(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            normalized = _normalize_type(item)
            if normalized:
                return normalized
        return ""
    return str(value or "").strip().lower()


def _recipe_nodes_from_json_ld(html: str) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for match in _JSON_LD_RE.finditer(html or ""):
        raw = (match.group(1) or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        queue: list[Any] = [payload]
        while queue:
            item = queue.pop(0)
            if isinstance(item, list):
                queue.extend(item)
                continue
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                queue.extend(graph)
            item_type = _normalize_type(item.get("@type"))
            if item_type == "recipe" or item_type.endswith("recipe"):
                nodes.append(item)
    return nodes


def _recipe_confidence(html: str) -> float:
    if _recipe_nodes_from_json_ld(html):
        return RECIPE_CONFIDENCE
    if re.search(r"itemtype=[\"']https?://schema.org/Recipe[\"']", html or "", re.I):
        return 0.92
    if _RECIPE_TYPE_RE.search(html or "") and "recipeingredient" in html.lower():
        return 0.85
    return 0.0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = html_module.unescape(value).strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_string_list(item))
        return out
    if isinstance(value, dict):
        text = str(value.get("text") or value.get("name") or "").strip()
        return [text] if text else []
    return []


def _instructions_from_recipe(recipe: dict[str, Any]) -> list[str]:
    instructions = recipe.get("recipeInstructions") or recipe.get("instructions") or []
    if isinstance(instructions, str):
        return [instructions.strip()] if instructions.strip() else []
    steps: list[str] = []
    if isinstance(instructions, list):
        for item in instructions:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("name") or "").strip()
                if text:
                    steps.append(text)
            elif isinstance(item, str) and item.strip():
                steps.append(item.strip())
    return steps


def _document_from_recipe_dict(recipe: dict[str, Any], *, url: str, fetch_tier: str) -> Document:
    title = str(recipe.get("name") or recipe.get("headline") or "").strip() or None
    ingredients = _string_list(recipe.get("recipeIngredient") or recipe.get("ingredients"))
    steps = _instructions_from_recipe(recipe)
    description = str(recipe.get("description") or "").strip()

    sections: list[DocumentSection] = []
    if description:
        sections.append(DocumentSection(heading="Description", level=2, text=description))
    if ingredients:
        sections.append(
            DocumentSection(
                heading="Ingredients",
                level=2,
                text="",
                list_items=tuple(ingredients),
            )
        )
    if steps:
        sections.append(
            DocumentSection(
                heading="Instructions",
                level=2,
                text="\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, start=1)),
                list_items=tuple(steps),
            )
        )
    if not sections and title:
        sections.append(DocumentSection(heading=title, level=1, text=title))

    structured = {
        "type": "Recipe",
        "name": title,
        "ingredients": ingredients,
        "instructions": steps,
        "yield": recipe.get("recipeYield") or recipe.get("yield"),
        "total_time": recipe.get("totalTime"),
    }

    return Document(
        url=url,
        title=title,
        author=_string_list(recipe.get("author"))[:1][0] if _string_list(recipe.get("author")) else None,
        date=str(recipe.get("datePublished") or "").strip() or None,
        sections=sections,
        structured_data=structured,
        metadata=DocumentMetadata(
            extractor_name=EXTRACTOR_NAME,
            extractor_version=EXTRACTOR_VERSION,
            extractor_confidence=RECIPE_CONFIDENCE,
            fetch_tier=fetch_tier,
        ),
    )


def _document_from_scraper(html: str, url: str, *, fetch_tier: str) -> Document | None:
    if scrape_html is None:
        return None
    try:
        scraper = scrape_html(html, org_url=url, online=False)
    except Exception:
        return None

    ingredients = [str(x).strip() for x in (scraper.ingredients() or []) if str(x).strip()]
    steps = [str(x).strip() for x in (scraper.instructions_list() or []) if str(x).strip()]
    title = str(scraper.title() or "").strip() or None
    sections: list[DocumentSection] = []
    if ingredients:
        sections.append(
            DocumentSection(
                heading="Ingredients",
                level=2,
                text="",
                list_items=tuple(ingredients),
            )
        )
    if steps:
        sections.append(
            DocumentSection(
                heading="Instructions",
                level=2,
                text="\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, start=1)),
                list_items=tuple(steps),
            )
        )
    if not sections:
        return None

    return Document(
        url=url,
        title=title,
        sections=sections,
        structured_data={
            "type": "Recipe",
            "name": title,
            "ingredients": ingredients,
            "instructions": steps,
            "yield": scraper.yields(),
            "total_time": scraper.total_time(),
        },
        metadata=DocumentMetadata(
            extractor_name=EXTRACTOR_NAME,
            extractor_version=EXTRACTOR_VERSION,
            extractor_confidence=RECIPE_CONFIDENCE,
            fetch_tier=fetch_tier,
        ),
    )


class RecipeExtractor:
    metadata = ExtractorMetadata(
        name=EXTRACTOR_NAME,
        version=EXTRACTOR_VERSION,
        priority=90,
    )

    def supports(
        self,
        url: str,
        html: str,
        *,
        headers=None,
    ) -> float:
        _ = (url, headers)
        return _recipe_confidence(html)

    def extract(
        self,
        html: str,
        url: str,
        *,
        fetch_tier: str = "http",
    ) -> Document:
        recipes = _recipe_nodes_from_json_ld(html)
        if recipes:
            return _document_from_recipe_dict(recipes[0], url=url, fetch_tier=fetch_tier)

        scraped = _document_from_scraper(html, url, fetch_tier=fetch_tier)
        if scraped is not None:
            return scraped

        return Document(
            url=url,
            title=None,
            sections=[],
            structured_data={"type": "Recipe"},
            metadata=DocumentMetadata(
                extractor_name=EXTRACTOR_NAME,
                extractor_version=EXTRACTOR_VERSION,
                extractor_confidence=RECIPE_CONFIDENCE,
                fetch_tier=fetch_tier,
            ),
        )
