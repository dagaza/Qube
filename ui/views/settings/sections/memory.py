"""Memory settings section."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import (
    get_enable_memory_consolidation,
    get_enable_memory_enrichment,
    get_enable_memory_promotion,
)
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle
from ui.views.settings.widgets import add_subsection_to_form


def build_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(container)
    layout.setSpacing(15)

    # --- Memory pipeline ---
    memory_widget = QWidget()
    memory_form = QFormLayout(memory_widget)
    memory_form.setSpacing(15)
    memory_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    add_subsection_to_form(memory_form, "Memory pipeline", anchor="memory")

    host.memory_enrichment_toggle = PrestigeToggle()
    host.mem_enrichment_label = QLabel(
        "Enable Memory Enrichment & Reflection (Requires more RAM)"
    )
    host.mem_enrichment_label.setWordWrap(True)
    _mem_enrichment_tip = (
        "When enabled, Qube extracts durable facts from chat, summarises sessions "
        "into episodic memories, and runs a periodic LLM audit that flags suspicious "
        "stored memories for review. Uses more RAM and background LLM time. "
        "When disabled, existing memories and retrieval still work; usage counters "
        "and decay maintenance for stored rows continue."
    )
    host.memory_enrichment_toggle.setToolTip(_mem_enrichment_tip)
    host.mem_enrichment_label.setToolTip(_mem_enrichment_tip)
    mem_row = QWidget()
    mem_row_layout = QHBoxLayout(mem_row)
    mem_row_layout.setContentsMargins(0, 0, 0, 0)
    mem_row_layout.addWidget(
        host.memory_enrichment_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    mem_row_layout.addWidget(host.mem_enrichment_label, stretch=1)

    host.memory_enrichment_toggle.blockSignals(True)
    host.memory_enrichment_toggle.setChecked(get_enable_memory_enrichment())
    host.memory_enrichment_toggle.blockSignals(False)
    host.memory_enrichment_toggle.toggled.connect(host._on_memory_enrichment_toggled)

    host.memory_promotion_toggle = PrestigeToggle()
    host.mem_promotion_label = QLabel("Promote well-used memories to preferences")
    host.mem_promotion_label.setWordWrap(True)
    _mem_promotion_tip = (
        "When this is on, Qube occasionally upgrades facts you rely on often into "
        "long-term preferences — the kind of thing Qube should remember about you "
        "without being asked each time.\n\n"
        "It looks at how often a memory is recalled in chat, whether answers actually "
        "use it, and whether it comes up in different conversations. Requires "
        "Memory Enrichment above.\n\n"
        "Off by default. Qube never removes memories on its own — you can always "
        "review or edit everything in Memory Manager."
    )
    host.memory_promotion_toggle.setToolTip(_mem_promotion_tip)
    host.mem_promotion_label.setToolTip(_mem_promotion_tip)
    promo_row = QWidget()
    promo_row_layout = QHBoxLayout(promo_row)
    promo_row_layout.setContentsMargins(0, 0, 0, 0)
    promo_row_layout.addWidget(
        host.memory_promotion_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    promo_row_layout.addWidget(host.mem_promotion_label, stretch=1)
    host.memory_promotion_toggle.blockSignals(True)
    host.memory_promotion_toggle.setChecked(get_enable_memory_promotion())
    host.memory_promotion_toggle.blockSignals(False)
    host.memory_promotion_toggle.toggled.connect(host._on_memory_promotion_toggled)

    host.memory_promotion_preset_selector = SelectorButton("Standard", is_dark=is_dark)
    host.memory_promotion_preset_selector.setMinimumWidth(200)
    host.memory_promotion_preset_selector.setMaximumWidth(280)
    host.memory_promotion_preset_selector.setToolTip(
        "How cautious Qube should be before promoting a memory.\n\n"
        "Conservative — waits for more repeated use before upgrading.\n"
        "Standard — balanced default.\n"
        "Aggressive — promotes sooner.\n\n"
        "Only applies when Promote well-used memories is enabled."
    )

    host.memory_consolidation_toggle = PrestigeToggle()
    host.mem_consolidation_label = QLabel("Highlight memories that keep coming back")
    host.mem_consolidation_label.setWordWrap(True)
    _mem_consolidation_tip = (
        "When this is on, Qube watches for memories that show up again on "
        "different days — a hint they may matter more than one-off notes.\n\n"
        "Those items get a gentle nudge in Memory Manager (marked for your "
        "review). Qube does not rewrite or delete them automatically, and "
        "this runs quietly in the background.\n\n"
        "On by default. Turn off if you prefer to curate memories only yourself."
    )
    host.memory_consolidation_toggle.setToolTip(_mem_consolidation_tip)
    host.mem_consolidation_label.setToolTip(_mem_consolidation_tip)
    consolidate_row = QWidget()
    consolidate_row_layout = QHBoxLayout(consolidate_row)
    consolidate_row_layout.setContentsMargins(0, 0, 0, 0)
    consolidate_row_layout.addWidget(
        host.memory_consolidation_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    consolidate_row_layout.addWidget(host.mem_consolidation_label, stretch=1)
    host.memory_consolidation_toggle.blockSignals(True)
    host.memory_consolidation_toggle.setChecked(get_enable_memory_consolidation())
    host.memory_consolidation_toggle.blockSignals(False)
    host.memory_consolidation_toggle.toggled.connect(
        host._on_memory_consolidation_toggled
    )

    promo_preset_row = QWidget()
    promo_preset_layout = QHBoxLayout(promo_preset_row)
    promo_preset_layout.setContentsMargins(0, 0, 0, 0)
    host._promo_preset_lbl = QLabel("Promotion preset")
    host._promo_preset_lbl.setToolTip(host.memory_promotion_preset_selector.toolTip())
    promo_preset_layout.addWidget(host._promo_preset_lbl)
    promo_preset_layout.addWidget(host.memory_promotion_preset_selector)
    promo_preset_layout.addStretch(1)

    memory_form.addRow("", mem_row)
    memory_form.addRow("", promo_row)
    memory_form.addRow("", promo_preset_row)
    memory_form.addRow("", consolidate_row)
    layout.addWidget(memory_widget)

    # --- Personalization ---
    personal_widget = QWidget()
    personal_form = QFormLayout(personal_widget)
    personal_form.setSpacing(15)
    personal_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    add_subsection_to_form(personal_form, "Personalization", anchor="personalization")

    host.profile_units_selector = SelectorButton("Use inferred units", is_dark=is_dark)
    host.profile_units_selector.setMinimumWidth(200)
    host.profile_units_selector.setMaximumWidth(280)
    host.profile_units_selector.setToolTip(
        "Default measurement units for weather and other numeric answers. "
        "Unset lets Qube learn units from conversation."
    )
    profile_units_row = QWidget()
    profile_units_layout = QHBoxLayout(profile_units_row)
    profile_units_layout.setContentsMargins(0, 0, 0, 0)
    profile_units_lbl = QLabel("Default units")
    profile_units_lbl.setToolTip(host.profile_units_selector.toolTip())
    profile_units_layout.addWidget(profile_units_lbl)
    profile_units_layout.addWidget(host.profile_units_selector)
    profile_units_layout.addStretch(1)
    personal_form.addRow("", profile_units_row)
    layout.addWidget(personal_widget)

    host._build_memory_promotion_preset_menu()
    host._build_profile_units_menu()
    host._sync_memory_promotion_controls_for_enrichment()
    host._sync_profile_units_selector()

    return container
