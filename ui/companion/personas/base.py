"""Base protocol for companion persona renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PyQt6.QtGui import QPainter

from core.companion_personas import CompanionPersonaId
from ui.companion.persona_context import CompanionPaintContext


class CompanionPersonaRenderer(ABC):
    """Persona-specific paint implementation."""

    persona_id: CompanionPersonaId

    @abstractmethod
    def paint(self, painter: QPainter, ctx: CompanionPaintContext) -> None:
        """Draw the persona centered at ctx.center_x / ctx.center_y."""

    @abstractmethod
    def halo_extra_px(self, body_radius: float) -> int:
        """Extra margin beyond body for waveform / aura (layout sizing)."""

    def visual_extent_px(self, body_radius: float) -> float:
        """Max distance from center to furthest painted pixel (incl. float drift)."""
        return body_radius + self.halo_extra_px(body_radius) + 4.0


def get_persona_renderer(persona_id: CompanionPersonaId | str) -> CompanionPersonaRenderer:
    from core.companion_cube_style import CompanionCubeStyle, normalize_companion_cube_style
    from core.companion_personas import normalize_companion_persona
    from core import app_settings
    from ui.companion.personas.qube_cube_classic import QubeCubeClassicPersonaRenderer
    from ui.companion.personas.qube_cube_experimental import QubeCubeExperimentalPersonaRenderer
    from ui.companion.personas.sphere import SpherePersonaRenderer

    resolved = normalize_companion_persona(
        persona_id.value if isinstance(persona_id, CompanionPersonaId) else persona_id
    )
    if resolved == CompanionPersonaId.QUBE:
        style = normalize_companion_cube_style(app_settings.get_companion_cube_style())
        if style == CompanionCubeStyle.CLASSIC:
            return QubeCubeClassicPersonaRenderer()
        return QubeCubeExperimentalPersonaRenderer()
    return SpherePersonaRenderer()
