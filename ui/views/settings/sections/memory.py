"""Memory settings section."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import (
    get_advanced_memory_unlocked,
    get_enable_memory_consolidation,
    get_enable_memory_enrichment,
    get_enable_memory_promotion,
)
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_subsection_to_form,
    add_section_reset_footer,
    add_settings_card_form,
    add_settings_full_width_row,
    make_settings_form,
    prepare_settings_card_form,
    register_settings_selector_width,
    schedule_settings_selector_refit,
)


def build_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(container)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    # --- Memory pipeline (simple / everyday) ---
    pipeline_card, pipeline_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    pipeline_form_host, pipeline_form = prepare_settings_card_form(pipeline_card_layout)
    add_subsection_to_form(pipeline_form, "Memory pipeline", anchor="memory")

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

    add_settings_full_width_row(pipeline_form, mem_row)
    pipeline_card_layout.addWidget(pipeline_form_host)
    layout.addWidget(pipeline_card)

    # --- Advanced memory card (promotion / consolidation) ---
    advanced_card, advanced_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    advanced_form = add_settings_card_form(advanced_card_layout)
    add_subsection_to_form(
        advanced_form, "Advanced memory", anchor="advanced_memory"
    )

    _adv_memory_tip = (
        "Advanced memory controls are not for everyday use.\n\n"
        "Unlocks optional promotion and consolidation workers that adjust how "
        "memories graduate into preferences and how recurring themes are highlighted "
        "in Memory Manager.\n\n"
        "Promotion and consolidation stay off until you enable them here."
    )
    host.advanced_memory_toggle = PrestigeToggle()
    host.advanced_memory_toggle.setToolTip(_adv_memory_tip)
    host.advanced_memory_label = QLabel("Show advanced memory settings")
    host.advanced_memory_label.setToolTip(_adv_memory_tip)
    host.advanced_memory_info_btn = host._make_settings_info_button(_adv_memory_tip)
    adv_label_cluster = QWidget()
    adv_label_layout = QHBoxLayout(adv_label_cluster)
    adv_label_layout.setContentsMargins(0, 0, 0, 0)
    adv_label_layout.setSpacing(6)
    adv_label_layout.addWidget(host.advanced_memory_label)
    adv_label_layout.addWidget(host.advanced_memory_info_btn)
    advanced_row = QWidget()
    advanced_row_layout = QHBoxLayout(advanced_row)
    advanced_row_layout.setContentsMargins(0, 0, 0, 0)
    advanced_row_layout.setSpacing(8)
    advanced_row_layout.addWidget(
        host.advanced_memory_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    advanced_row_layout.addWidget(adv_label_cluster)
    advanced_row_layout.addStretch(1)
    host.advanced_memory_toggle.blockSignals(True)
    host.advanced_memory_toggle.setChecked(get_advanced_memory_unlocked())
    host.advanced_memory_toggle.blockSignals(False)
    host.advanced_memory_toggle.toggled.connect(host._on_advanced_memory_toggled)
    add_settings_full_width_row(advanced_form, advanced_row)

    host.advanced_memory_panel = QWidget()
    adv_panel_layout = QVBoxLayout(host.advanced_memory_panel)
    adv_panel_layout.setContentsMargins(0, 8, 0, 0)
    adv_panel_layout.setSpacing(12)

    adv_form_host, adv_form = make_settings_form()

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
    register_settings_selector_width(
        host.memory_promotion_preset_selector,
        "Conservative",
        "Standard",
        "Aggressive",
    )
    host.memory_promotion_preset_selector.setMenu(QMenu(host.memory_promotion_preset_selector))
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
        "Off by default. Turn on if you want recurring-theme hints in Memory Manager."
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

    add_settings_full_width_row(adv_form, promo_row)
    add_settings_full_width_row(adv_form, promo_preset_row)
    add_settings_full_width_row(adv_form, consolidate_row)
    adv_panel_layout.addWidget(adv_form_host)
    host.advanced_memory_panel.setVisible(get_advanced_memory_unlocked())
    add_settings_full_width_row(advanced_form, host.advanced_memory_panel)
    layout.addWidget(advanced_card)

    host._build_memory_promotion_preset_menu()
    host._sync_memory_promotion_controls_for_enrichment()

    for attr in ("memory_promotion_preset_selector",):
        selector = getattr(host, attr, None)
        if selector is not None:
            schedule_settings_selector_refit(selector)

    add_section_reset_footer(layout, host, "memory", is_dark=is_dark)

    return container
