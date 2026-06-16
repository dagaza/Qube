"""Voice & Audio settings section."""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_advanced_stt_unlocked, get_advanced_tts_unlocked
from core.stt_models import get_stt_models_dir
from core.tts_models import get_tts_models_dir
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle
from ui.views.settings.controls import NoScrollDoubleSpinBox, NoScrollSpinBox
from ui.views.settings.widgets import (
    add_section_divider_to_form,
    add_section_reset_footer,
    add_subsection_to_form,
    wrap_subsection,
)

_DEVICE_SELECTOR_WIDTH = 350
_WAKEWORD_ACTION_BTN_WIDTH = 300


def _apply_device_selector_width(selector: SelectorButton) -> None:
    selector.setFixedWidth(_DEVICE_SELECTOR_WIDTH)


def _apply_wakeword_action_button_width(btn: QPushButton) -> None:
    """Keep wakeword download + test-lab buttons the same compact width."""
    btn.setFixedWidth(_WAKEWORD_ACTION_BTN_WIDTH)
    policy = btn.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
    policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
    btn.setSizePolicy(policy)


def _preview_play_button(host, *, tooltip: str, handler) -> QPushButton:
    btn = QPushButton()
    btn.setObjectName("TtsVoicePreviewButton")
    btn.setFixedSize(32, 32)
    btn.setIcon(qta.icon("fa5s.play", color="#64748b"))
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.clicked.connect(handler)
    return btn


def _hint_button(host, *, tooltip: str, handler) -> QPushButton:
    btn = QPushButton()
    btn.setObjectName("AudioInputHintButton")
    btn.setFixedSize(32, 32)
    btn.setIcon(qta.icon("fa5s.lightbulb", color="#64748b"))
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.clicked.connect(handler)
    return btn


def _selector_action_row(selector: SelectorButton, *action_buttons: QPushButton) -> QWidget:
    """Keep selector width aligned with other device dropdowns; optional trailing actions."""
    _apply_device_selector_width(selector)
    if not action_buttons:
        return selector
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(selector, alignment=Qt.AlignmentFlag.AlignLeft)
    for btn in action_buttons:
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)
    layout.addStretch(1)
    return row


def _device_selector_row(selector: SelectorButton, preview_btn: QPushButton | None = None) -> QWidget:
    """Keep selector width aligned with other device dropdowns; optional preview control."""
    if preview_btn is None:
        _apply_device_selector_width(selector)
        return selector
    return _selector_action_row(selector, preview_btn)


def _advanced_toggle_row(
    host,
    *,
    label_text: str,
    tooltip: str,
    toggle_attr: str,
    label_attr: str,
    info_attr: str,
    handler_name: str,
    initially_unlocked: bool,
) -> QWidget:
    toggle = PrestigeToggle()
    toggle.setToolTip(tooltip)
    label = QLabel(label_text)
    label.setToolTip(tooltip)
    info_btn = host._make_settings_info_button(tooltip)
    label_cluster = QWidget()
    label_layout = QHBoxLayout(label_cluster)
    label_layout.setContentsMargins(0, 0, 0, 0)
    label_layout.setSpacing(6)
    label_layout.addWidget(label)
    label_layout.addWidget(info_btn)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    row_layout.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignLeft)
    row_layout.addWidget(label_cluster)
    row_layout.addStretch(1)
    toggle.blockSignals(True)
    toggle.setChecked(initially_unlocked)
    toggle.blockSignals(False)
    toggle.toggled.connect(getattr(host, handler_name))
    setattr(host, toggle_attr, toggle)
    setattr(host, label_attr, label)
    setattr(host, info_attr, info_btn)
    return row


def _add_stt_advanced_options(host, form: QFormLayout) -> None:
    add_subsection_to_form(form, "Speech-to-text (STT)", anchor="stt_models")

    _stt_adv_tip = (
        "Advanced STT controls are not for everyday use.\n\n"
        "Unlocks optional speech-to-text model selection. Place CTranslate2 Whisper "
        "folders (each must contain model.bin) under models/stt/, then select one here.\n\n"
        "The bundled Whisper small default cannot be deleted."
    )
    form.addRow(
        "",
        _advanced_toggle_row(
            host,
            label_text="Show advanced STT settings",
            tooltip=_stt_adv_tip,
            toggle_attr="advanced_stt_toggle",
            label_attr="advanced_stt_label",
            info_attr="advanced_stt_info_btn",
            handler_name="_on_advanced_stt_toggled",
            initially_unlocked=get_advanced_stt_unlocked(),
        ),
    )

    host.advanced_stt_panel = QWidget()
    stt_panel_layout = QVBoxLayout(host.advanced_stt_panel)
    stt_panel_layout.setContentsMargins(0, 8, 0, 0)
    stt_panel_layout.setSpacing(12)

    stt_inner = QWidget()
    stt_form = QFormLayout(stt_inner)
    stt_form.setSpacing(15)
    stt_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.stt_dir_label = QLabel(get_stt_models_dir())
    host.stt_dir_label.setWordWrap(True)
    host.stt_dir_label.setToolTip(
        "Bundled Whisper small downloads here on first use. Place custom CTranslate2 "
        "Whisper folders alongside (each needs model.bin)."
    )

    stt_row = QHBoxLayout()
    host.stt_model_list = QListWidget()
    host.stt_model_list.setMinimumHeight(80)
    host.stt_model_list.setMaximumHeight(120)
    host.stt_model_list.setToolTip(
        "Select a speech-to-text model, then click Use selected."
    )
    stt_row.addWidget(host.stt_model_list, stretch=1)
    stt_btn_col = QVBoxLayout()
    stt_btn_col.setSpacing(8)
    host.use_stt_model_btn = QPushButton("Use selected")
    apply_brand_primary(host.use_stt_model_btn)
    host.use_stt_model_btn.clicked.connect(host._apply_selected_stt_model)
    stt_btn_col.addWidget(host.use_stt_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.reset_stt_model_btn = QPushButton("Reset to default")
    apply_brand_primary(host.reset_stt_model_btn, icon_name="fa5s.undo")
    host.reset_stt_model_btn.clicked.connect(host._reset_stt_to_default)
    stt_btn_col.addWidget(host.reset_stt_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.refresh_stt_model_btn = QPushButton("Refresh")
    host.refresh_stt_model_btn.clicked.connect(host._on_refresh_stt_models_clicked)
    stt_btn_col.addWidget(host.refresh_stt_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.delete_stt_model_btn = QPushButton("Delete")
    apply_brand_danger(host.delete_stt_model_btn)
    host.delete_stt_model_btn.clicked.connect(host._delete_selected_stt_model)
    stt_btn_col.addWidget(host.delete_stt_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    stt_row.addLayout(stt_btn_col)

    host.active_stt_model_lbl = QLabel()
    host.active_stt_model_lbl.setWordWrap(True)

    stt_form.addRow("Model storage", host.stt_dir_label)
    stt_form.addRow("On this device", stt_row)
    stt_form.addRow("Active model", host.active_stt_model_lbl)

    stt_panel_layout.addWidget(wrap_subsection(stt_inner, anchor="stt_models"))
    host.advanced_stt_panel.setVisible(get_advanced_stt_unlocked())
    form.addRow("", host.advanced_stt_panel)


def _add_tts_advanced_options(host, form: QFormLayout) -> None:
    add_subsection_to_form(form, "Text-to-speech (TTS)", anchor="tts_models")

    _tts_adv_tip = (
        "Advanced TTS controls are not for everyday use.\n\n"
        "Unlocks optional text-to-speech model selection. Place .onnx files under "
        "models/tts/ (Kokoro also needs voices-v1.0.bin in the same folder), then "
        "select one here.\n\n"
        "The bundled Kokoro v1.0 default cannot be deleted."
    )
    form.addRow(
        "",
        _advanced_toggle_row(
            host,
            label_text="Show advanced TTS settings",
            tooltip=_tts_adv_tip,
            toggle_attr="advanced_tts_toggle",
            label_attr="advanced_tts_label",
            info_attr="advanced_tts_info_btn",
            handler_name="_on_advanced_tts_toggled",
            initially_unlocked=get_advanced_tts_unlocked(),
        ),
    )

    host.advanced_tts_panel = QWidget()
    tts_panel_layout = QVBoxLayout(host.advanced_tts_panel)
    tts_panel_layout.setContentsMargins(0, 8, 0, 0)
    tts_panel_layout.setSpacing(12)

    tts_inner = QWidget()
    tts_form = QFormLayout(tts_inner)
    tts_form.setSpacing(15)
    tts_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.tts_dir_label = QLabel(get_tts_models_dir())
    host.tts_dir_label.setWordWrap(True)
    host.tts_dir_label.setToolTip(
        "Bundled Kokoro v1.0 lives here. Place optional .onnx TTS models in this folder."
    )

    tts_row = QHBoxLayout()
    host.tts_model_list = QListWidget()
    host.tts_model_list.setMinimumHeight(80)
    host.tts_model_list.setMaximumHeight(120)
    host.tts_model_list.setToolTip(
        "Select a text-to-speech model, then click Use selected."
    )
    tts_row.addWidget(host.tts_model_list, stretch=1)
    tts_btn_col = QVBoxLayout()
    tts_btn_col.setSpacing(8)
    host.use_tts_model_btn = QPushButton("Use selected")
    apply_brand_primary(host.use_tts_model_btn)
    host.use_tts_model_btn.clicked.connect(host._apply_selected_tts_model)
    tts_btn_col.addWidget(host.use_tts_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.reset_tts_model_btn = QPushButton("Reset to default")
    apply_brand_primary(host.reset_tts_model_btn, icon_name="fa5s.undo")
    host.reset_tts_model_btn.clicked.connect(host._reset_tts_to_default)
    tts_btn_col.addWidget(host.reset_tts_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.refresh_tts_model_btn = QPushButton("Refresh")
    host.refresh_tts_model_btn.clicked.connect(host._on_refresh_tts_models_clicked)
    tts_btn_col.addWidget(host.refresh_tts_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.delete_tts_model_btn = QPushButton("Delete")
    apply_brand_danger(host.delete_tts_model_btn)
    host.delete_tts_model_btn.clicked.connect(host._delete_selected_tts_model)
    tts_btn_col.addWidget(host.delete_tts_model_btn, alignment=Qt.AlignmentFlag.AlignTop)
    tts_row.addLayout(tts_btn_col)

    host.active_tts_model_lbl = QLabel()
    host.active_tts_model_lbl.setWordWrap(True)

    tts_form.addRow("Model storage", host.tts_dir_label)
    tts_form.addRow("On this device", tts_row)
    tts_form.addRow("Active model", host.active_tts_model_lbl)

    tts_panel_layout.addWidget(wrap_subsection(tts_inner, anchor="tts_models"))
    host.advanced_tts_panel.setVisible(get_advanced_tts_unlocked())
    form.addRow("", host.advanced_tts_panel)


def build_section(host, *, is_dark: bool) -> QWidget:
    """Build the Voice & Audio settings form; widgets are attached to ``host``."""
    section_widget = QWidget()
    section_widget.setObjectName("SettingsFormContainer")

    form = QFormLayout(section_widget)
    form.setSpacing(15)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    add_subsection_to_form(form, "Devices")

    host.mic_selector = SelectorButton("Select Input Device...", is_dark=is_dark)
    host.device_selector = SelectorButton("Select Output Device...", is_dark=is_dark)
    host.voice_selector = SelectorButton("Select Voice...", is_dark=is_dark)
    host._tts_voice_preview_phrase_index = 0

    for btn in (host.mic_selector, host.device_selector, host.voice_selector):
        _apply_device_selector_width(btn)
        btn.setMenu(QMenu(btn))

    host.mic_selector.setToolTip(
        "Microphone used for voice input and wakeword detection."
    )
    host.device_selector.setToolTip(
        "Speaker or headset used for text-to-speech playback."
    )
    host.voice_selector.setToolTip("Default text-to-speech voice for spoken responses.")

    host.audio_output_preview_btn = _preview_play_button(
        host,
        tooltip=(
            "Play a short sample on the selected output device using the current TTS voice."
        ),
        handler=host._play_tts_voice_preview,
    )
    host.tts_voice_preview_btn = _preview_play_button(
        host,
        tooltip="Play a short sample with the selected voice.",
        handler=host._play_tts_voice_preview,
    )
    host.audio_input_hint_btn = _hint_button(
        host,
        tooltip=(
            "Highlight the microphone level meter in the top bar. Speak to confirm "
            "your mic is picking up sound on the selected input."
        ),
        handler=host._on_audio_input_hint_clicked,
    )

    form.addRow(
        "Audio Input",
        _selector_action_row(host.mic_selector, host.audio_input_hint_btn),
    )
    form.addRow(
        "Audio Output",
        _device_selector_row(host.device_selector, host.audio_output_preview_btn),
    )
    form.addRow(
        "TTS Voice",
        _device_selector_row(host.voice_selector, host.tts_voice_preview_btn),
    )

    add_subsection_to_form(form, "Wakeword", anchor="wakeword")

    host.wakeword_selector = SelectorButton("Select Wakeword...", is_dark=is_dark)
    host.wakeword_selector.setMenu(QMenu(host.wakeword_selector))
    host.wakeword_selector.setFixedWidth(300)
    host.wakeword_selector.setToolTip(
        "Always run Wakeword Testbed after selecting a wakeword. "
        "Both Community and Recommended wakewords can perform differently "
        "depending on your voice, mic setup, room noise, and sensitivity."
        "You can always download your own wakewords and place them in the wakewords folder."
    )

    wakeword_row = QWidget()
    wakeword_row_layout = QHBoxLayout(wakeword_row)
    wakeword_row_layout.setContentsMargins(0, 0, 0, 0)
    wakeword_row_layout.setSpacing(8)
    wakeword_row_layout.addWidget(host.wakeword_selector)

    host.wakeword_info_btn = QPushButton()
    host.wakeword_info_btn.setFixedSize(24, 24)
    host.wakeword_info_btn.setObjectName("WakewordInfoButton")
    host.wakeword_info_btn.setIcon(qta.icon("fa5s.info-circle", color="#64748b"))
    host.wakeword_info_btn.setToolTip(host.wakeword_selector.toolTip())
    host.wakeword_info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    wakeword_row_layout.addWidget(host.wakeword_info_btn)
    wakeword_row_layout.addStretch()

    form.addRow("Active Wakeword", wakeword_row)

    host.wakeword_download_open_btn = QPushButton("Download OpenWakeWord models")
    apply_brand_primary(host.wakeword_download_open_btn)
    _apply_wakeword_action_button_width(host.wakeword_download_open_btn)
    host.wakeword_download_open_btn.setToolTip(
        "Downloads OpenWakeWord wakeword models (built-in set) and the required feature assets."
    )

    host.wakeword_download_community_btn = QPushButton("Download Community models")
    apply_brand_primary(host.wakeword_download_community_btn)
    _apply_wakeword_action_button_width(host.wakeword_download_community_btn)
    host.wakeword_download_community_btn.setToolTip(
        "Downloads the community wakeword pack into your local wakeword folder."
    )

    wakeword_download_col = QWidget()
    wakeword_download_layout = QVBoxLayout(wakeword_download_col)
    wakeword_download_layout.setContentsMargins(0, 0, 0, 0)
    wakeword_download_layout.setSpacing(8)
    wakeword_download_layout.addWidget(
        host.wakeword_download_open_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )
    wakeword_download_layout.addWidget(
        host.wakeword_download_community_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )
    wakeword_download_layout.addStretch(1)
    form.addRow("", wakeword_download_col)

    host.wakeword_download_open_btn.clicked.connect(host._download_openwakeword_models)
    host.wakeword_download_community_btn.clicked.connect(host._download_community_wakeword_models)

    host.wakeword_test_lab_btn = QPushButton("Open Wakeword Test Lab")
    apply_brand_primary(host.wakeword_test_lab_btn)
    _apply_wakeword_action_button_width(host.wakeword_test_lab_btn)
    host.wakeword_test_lab_btn.setToolTip(
        "Test wakeword detection with your microphone before relying on it in conversation."
    )
    host.wakeword_test_lab_btn.clicked.connect(host._open_wakeword_test_lab)

    wakeword_lab_row = QWidget()
    wakeword_lab_layout = QHBoxLayout(wakeword_lab_row)
    wakeword_lab_layout.setContentsMargins(0, 0, 0, 0)
    wakeword_lab_layout.addWidget(host.wakeword_test_lab_btn, 0)
    wakeword_lab_layout.addStretch(1)
    form.addRow("", wakeword_lab_row)

    add_subsection_to_form(form, "Speech Detection")

    host.timeout_spinner = NoScrollDoubleSpinBox()
    host.timeout_spinner.setFixedWidth(90)
    host.timeout_spinner.setRange(0.5, 5.0)
    host.timeout_spinner.setSingleStep(0.1)
    host.timeout_spinner.setValue(
        host.audio_worker.silence_timeout if host.audio_worker else 2.0
    )
    host.timeout_spinner.setSuffix(" sec")
    host.timeout_spinner.setToolTip(
        "The amount of silence (in seconds) the app waits before deciding you have finished speaking. "
        "Lower values make the app respond faster, but it might interrupt you if you pause to think."
    )
    if host.audio_worker:
        host.timeout_spinner.valueChanged.connect(host.audio_worker.set_silence_timeout)

    host.threshold_spinner = NoScrollSpinBox()
    host.threshold_spinner.setFixedWidth(90)
    host.threshold_spinner.setRange(1, 100)
    host.threshold_spinner.setValue(
        int(host.audio_worker.speech_threshold) if host.audio_worker else 2
    )
    host.threshold_spinner.setSuffix("%")
    host.threshold_spinner.setToolTip(
        "Controls how loud you must speak to trigger recording. A higher number acts as a "
        "stronger background noise filter, meaning you will need to speak louder to punch through. "
        "If you are in a quiet environment, use the lowest setting."
    )
    if host.audio_worker:
        host.threshold_spinner.valueChanged.connect(host.audio_worker.set_speech_threshold)

    form.addRow("Silence Cutoff", host.timeout_spinner)
    form.addRow("VAD Threshold", host.threshold_spinner)

    add_subsection_to_form(form, "Toolbar")

    host.pin_audio_cb = QCheckBox("Pin Audio Controls to Toolbar")
    host.pin_audio_cb.setToolTip(
        "When checked, Silence Cutoff and VAD Threshold appear in the right toolbar. "
        "Uncheck to hide them from the toolbar (settings still apply)."
    )
    host.pin_audio_cb.blockSignals(True)
    host.pin_audio_cb.setChecked(True)
    host.pin_audio_cb.blockSignals(False)
    host.pin_audio_cb.toggled.connect(host.audio_pin_toggle.emit)
    form.addRow("", host.pin_audio_cb)

    host.pin_tts_voice_cb = QCheckBox("Pin TTS Voice selector to Toolbar")
    host.pin_tts_voice_cb.setToolTip(
        "When checked, the TTS voice selector appears in the right toolbar. "
        "Uncheck to hide it from the toolbar (settings still apply)."
    )
    host.pin_tts_voice_cb.blockSignals(True)
    host.pin_tts_voice_cb.setChecked(True)
    host.pin_tts_voice_cb.blockSignals(False)
    host.pin_tts_voice_cb.toggled.connect(host.tts_voice_pin_toggle.emit)
    form.addRow("", host.pin_tts_voice_cb)

    host.voice_audio_section_divider = add_section_divider_to_form(form, is_dark=is_dark)
    add_subsection_to_form(form, "Advanced Voice & Audio Options")
    _add_stt_advanced_options(host, form)
    _add_tts_advanced_options(host, form)

    add_section_reset_footer(form, host, "voice.audio", is_dark=is_dark)

    return section_widget
