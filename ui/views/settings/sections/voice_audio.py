"""Voice & Audio settings section."""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from ui.components.brand_buttons import apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.views.settings.controls import NoScrollDoubleSpinBox, NoScrollSpinBox
from ui.views.settings.widgets import add_subsection_to_form


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

    for btn in (host.mic_selector, host.device_selector, host.voice_selector):
        btn.setMaximumWidth(350)
        btn.setMenu(QMenu(btn))

    host.mic_selector.setToolTip(
        "Microphone used for voice input and wakeword detection."
    )
    host.device_selector.setToolTip(
        "Speaker or headset used for text-to-speech playback."
    )
    host.voice_selector.setToolTip("Default text-to-speech voice for spoken responses.")

    form.addRow("Audio Input", host.mic_selector)
    form.addRow("Audio Output", host.device_selector)
    form.addRow("TTS Voice", host.voice_selector)

    add_subsection_to_form(form, "Wakeword")

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

    host.wakeword_test_lab_btn = QPushButton("Open Wakeword Test Lab")
    apply_brand_primary(host.wakeword_test_lab_btn)
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

    return section_widget
