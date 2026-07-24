"""Theme contrast validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.theme.color_utils import contrast_ratio
from core.theme.tokens import ResolvedTheme

WARN_CONTRAST = 4.5
BLOCK_CONTRAST = 3.0


@dataclass(frozen=True)
class ContrastCheck:
    label: str
    foreground: str
    background: str
    ratio: float

    @property
    def warns(self) -> bool:
        return self.ratio < WARN_CONTRAST

    @property
    def blocks(self) -> bool:
        return self.ratio < BLOCK_CONTRAST


@dataclass
class ThemeValidationResult:
    checks: list[ContrastCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def can_save(self) -> bool:
        return not any(check.blocks for check in self.checks)


class ThemeValidator:
    def __init__(
        self,
        *,
        warn_contrast: float = WARN_CONTRAST,
        block_contrast: float = BLOCK_CONTRAST,
    ) -> None:
        self._warn_contrast = warn_contrast
        self._block_contrast = block_contrast

    def validate(self, theme: ResolvedTheme) -> ThemeValidationResult:
        result = ThemeValidationResult()
        pairs = (
            ("Body text on canvas", theme.text_primary, theme.background),
            ("Body text on elevated surface", theme.text_primary, theme.surface_elevated),
            ("Text on accent", theme.text_on_accent, theme.accent),
        )
        for label, fg, bg in pairs:
            ratio = contrast_ratio(fg, bg)
            check = ContrastCheck(label=label, foreground=fg, background=bg, ratio=ratio)
            result.checks.append(check)
            if ratio < self._block_contrast:
                result.errors.append(
                    f"{label} contrast {ratio:.2f}:1 is below minimum "
                    f"{self._block_contrast:.1f}:1"
                )
            elif ratio < self._warn_contrast:
                result.warnings.append(
                    f"{label} contrast {ratio:.2f}:1 is below recommended "
                    f"{self._warn_contrast:.1f}:1"
                )
        return result
