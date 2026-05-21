"""Display-only labels for local .gguf files in toolbar and Settings pickers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.model_params import parse_params_b_from_filename
from core.app_settings import expected_gguf_shard_filenames, parse_gguf_shard_info
from core.gguf_quant import ParsedQuant, parse_quant_from_gguf_path

_SEP = " · "


@dataclass(frozen=True)
class LocalGgufDisplay:
    basename: str
    menu_label: str
    button_label: str
    tooltip: str


def _strip_quant_suffix(stem: str, parsed: ParsedQuant) -> str:
    token = re.escape(parsed.raw).replace(r"\-", "[-_.]").replace(r"\_", "[-_.]")
    pattern = re.compile(rf"[-_.]?{token}$", re.IGNORECASE)
    cleaned = pattern.sub("", stem).rstrip("-_.").strip()
    return cleaned or stem


def _shard_menu_label(path: str, models_dir: Path | None) -> str | None:
    shard_info = parse_gguf_shard_info(path)
    if shard_info is None:
        return None
    root = models_dir if models_dir is not None else Path(path).parent
    expected = expected_gguf_shard_filenames(path)
    found = [fname for fname in expected if (root / fname).is_file()]
    total = int(shard_info.get("total", len(expected)))
    bundle_name = f"{Path(str(shard_info['prefix'])).name}.gguf"
    return f"{bundle_name} ({len(found)}/{total} shards)"


def format_local_gguf_display(
    path: str,
    *,
    models_dir: str | Path | None = None,
) -> LocalGgufDisplay:
    """Build human-readable labels for a local GGUF without changing stored paths."""
    resolved = Path(str(path or "").strip()).resolve()
    basename = resolved.name
    tooltip = str(resolved)
    root = Path(models_dir) if models_dir is not None else resolved.parent

    shard_label = _shard_menu_label(str(resolved), root)
    if shard_label is not None:
        return LocalGgufDisplay(
            basename=basename,
            menu_label=shard_label,
            button_label=shard_label,
            tooltip=tooltip,
        )

    parsed = parse_quant_from_gguf_path(str(resolved))
    if parsed is not None:
        stem = basename[:-5] if basename.lower().endswith(".gguf") else basename
        model_stem = _strip_quant_suffix(stem, parsed)
        menu_label = f"{model_stem}{_SEP}{parsed.normalized}"
        return LocalGgufDisplay(
            basename=basename,
            menu_label=menu_label,
            button_label=menu_label,
            tooltip=tooltip,
        )

    return LocalGgufDisplay(
        basename=basename,
        menu_label=basename,
        button_label=basename,
        tooltip=tooltip,
    )


def local_gguf_sort_key(path: str | Path) -> tuple[float, str]:
    """Sort local models ascending by params-B; unknown sizes sort last, then by name."""
    p = Path(path)
    params = parse_params_b_from_filename(p.name)
    rank = params if params is not None else float("inf")
    return (rank, p.name.lower())
