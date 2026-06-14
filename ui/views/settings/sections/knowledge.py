"""Knowledge settings section."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_advanced_embedding_unlocked
from core.embedding_models import get_embedding_models_dir
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.toggle import PrestigeToggle
from ui.views.settings.widgets import add_subsection_to_layout, add_section_reset_footer, wrap_subsection


def build_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(container)
    layout.setSpacing(15)

    add_subsection_to_layout(layout, "Library search phrases", anchor="triggers")
    layout.addWidget(host._build_triggers_manager())

    add_subsection_to_layout(layout, "Embedding model", anchor="embedding_model")

    _adv_tip = (
        "Advanced embedding controls are not for everyday use.\n\n"
        "Unlocks optional embedding model selection for RAG and memory search. "
        "The bundled Nomic Embed v1.5 default cannot be deleted. Place alternate "
        ".gguf embedding models in the embedding folder, then select one here.\n\n"
        "Switching to a model with a different vector size will reset your vector "
        "index — re-ingest library documents afterward."
    )
    host.advanced_embedding_toggle = PrestigeToggle()
    host.advanced_embedding_label = QLabel("Show advanced embedding settings")
    host.advanced_embedding_toggle.setToolTip(_adv_tip)
    host.advanced_embedding_label.setToolTip(_adv_tip)
    host.advanced_embedding_info_btn = host._make_settings_info_button(_adv_tip)
    label_cluster = QWidget()
    label_cluster_layout = QHBoxLayout(label_cluster)
    label_cluster_layout.setContentsMargins(0, 0, 0, 0)
    label_cluster_layout.setSpacing(6)
    label_cluster_layout.addWidget(host.advanced_embedding_label)
    label_cluster_layout.addWidget(host.advanced_embedding_info_btn)
    advanced_row = QWidget()
    advanced_row_layout = QHBoxLayout(advanced_row)
    advanced_row_layout.setContentsMargins(0, 0, 0, 0)
    advanced_row_layout.setSpacing(8)
    advanced_row_layout.addWidget(
        host.advanced_embedding_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    advanced_row_layout.addWidget(label_cluster)
    advanced_row_layout.addStretch(1)
    host.advanced_embedding_toggle.blockSignals(True)
    host.advanced_embedding_toggle.setChecked(get_advanced_embedding_unlocked())
    host.advanced_embedding_toggle.blockSignals(False)
    host.advanced_embedding_toggle.toggled.connect(host._on_advanced_embedding_toggled)
    layout.addWidget(advanced_row)

    host.advanced_embedding_panel = QWidget()
    adv_panel_layout = QVBoxLayout(host.advanced_embedding_panel)
    adv_panel_layout.setContentsMargins(0, 8, 0, 0)
    adv_panel_layout.setSpacing(12)

    embedding_inner = QWidget()
    embedding_form = QFormLayout(embedding_inner)
    embedding_form.setSpacing(15)
    embedding_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.embedding_dir_label = QLabel(get_embedding_models_dir())
    host.embedding_dir_label.setWordWrap(True)
    host.embedding_dir_label.setToolTip(
        "Place optional embedding .gguf files here. The bundled Nomic default "
        "also lives in this folder."
    )

    embedding_row = QHBoxLayout()
    host.embedding_gguf_list = QListWidget()
    host.embedding_gguf_list.setMinimumHeight(90)
    host.embedding_gguf_list.setMaximumHeight(140)
    host.embedding_gguf_list.setToolTip(
        "Built-in Nomic Embed v1.5 default cannot be deleted. Select a custom model "
        "and click Use selected, or Reset to default."
    )
    embedding_row.addWidget(host.embedding_gguf_list, stretch=1)
    embedding_btn_col = QVBoxLayout()
    embedding_btn_col.setSpacing(8)
    host.use_embedding_gguf_btn = QPushButton("Use selected")
    apply_brand_primary(host.use_embedding_gguf_btn)
    host.use_embedding_gguf_btn.clicked.connect(host._apply_selected_embedding_gguf)
    embedding_btn_col.addWidget(
        host.use_embedding_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    host.reset_embedding_btn = QPushButton("Reset to default")
    apply_brand_primary(host.reset_embedding_btn, icon_name="fa5s.undo")
    host.reset_embedding_btn.clicked.connect(host._reset_embedding_to_default)
    embedding_btn_col.addWidget(
        host.reset_embedding_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    host.refresh_embedding_gguf_btn = QPushButton("Refresh")
    host.refresh_embedding_gguf_btn.setToolTip(
        "Rescan the embedding folder for .gguf files added while the app is running"
    )
    host.refresh_embedding_gguf_btn.clicked.connect(host._on_refresh_embedding_gguf_clicked)
    embedding_btn_col.addWidget(
        host.refresh_embedding_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    host.delete_embedding_gguf_btn = QPushButton("Delete")
    apply_brand_danger(host.delete_embedding_gguf_btn)
    host.delete_embedding_gguf_btn.clicked.connect(host._delete_selected_embedding_gguf)
    embedding_btn_col.addWidget(
        host.delete_embedding_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    embedding_row.addLayout(embedding_btn_col)

    host.active_embedding_model_lbl = QLabel()
    host.active_embedding_model_lbl.setWordWrap(True)

    embedding_form.addRow("Model storage", host.embedding_dir_label)
    embedding_form.addRow("On this device", embedding_row)
    embedding_form.addRow("Active model", host.active_embedding_model_lbl)

    adv_panel_layout.addWidget(wrap_subsection(embedding_inner, anchor="embedding_model"))
    host.advanced_embedding_panel.setVisible(get_advanced_embedding_unlocked())
    layout.addWidget(host.advanced_embedding_panel)

    add_section_reset_footer(layout, host, "knowledge", is_dark=is_dark)

    return container
