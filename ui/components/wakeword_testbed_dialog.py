import logging

import qtawesome as qta
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from core.wakeword_testbed import WakewordTestbedState
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.selector_button import SelectorButton
from ui.components.brand_buttons import (
    apply_brand_primary,
    apply_brand_success,
    apply_brand_danger,
)

logger = logging.getLogger("Qube.UI.WakewordTestbed")


class WakewordTestbedDialog(QDialog):
    def __init__(self, parent, audio_worker):
        super().__init__(parent)
        self.audio_worker = audio_worker
        self._is_dark = getattr(parent, "_is_dark_theme", True)
        self.setWindowTitle("Wakeword Test Lab")
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(740, 600)

        self._test_state = WakewordTestbedState()
        self._test_attempt_index = 0
        self._test_attempt_peak = 0.0
        self._test_phase = "ready"
        self._view_state = "ready"
        self._test_attempt_open = False
        self._test_attempt_mode = "idle"
        self._test_attempt_timeout_secs = 2
        self._test_attempt_seconds_left = 0
        self._false_positive_seconds_left = 0
        self._has_fresh_completed_test = False
        self._last_test_summary = None

        self._test_false_positive_timer = QTimer(self)
        self._test_false_positive_timer.setInterval(1000)
        self._test_false_positive_timer.timeout.connect(self._on_false_positive_tick)
        self._test_attempt_timer = QTimer(self)
        self._test_attempt_timer.setInterval(1000)
        self._test_attempt_timer.timeout.connect(self._on_attempt_tick)

        self._build_ui()
        self._apply_theme_styles()
        self._bind_signals()
        self._sync_wakeword_catalog()
        self._sync_sensitivity_from_worker()
        self._set_view_state("ready")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(0)

        self.container = QFrame()
        self.container.setObjectName("WakewordLabContainer")
        root = QVBoxLayout(self.container)
        self._root_layout = root
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        self.header_title_lbl = QLabel("Wakeword Test Lab")
        self.header_title_lbl.setObjectName("WakewordHeaderTitle")
        self.header_subtitle_lbl = QLabel(
            "Validate this wakeword before applying it to live detection."
        )
        self.header_subtitle_lbl.setObjectName("WakewordHeaderSubtitle")
        self.header_subtitle_lbl.setWordWrap(True)
        title_col.addWidget(self.header_title_lbl)
        title_col.addWidget(self.header_subtitle_lbl)

        self.header_close_btn = QPushButton("✕")
        self.header_close_btn.setObjectName("WakewordHeaderCloseButton")
        self.header_close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header_close_btn.setFixedSize(30, 30)
        header_row.addLayout(title_col, 1)
        header_row.addWidget(self.header_close_btn, 0, Qt.AlignmentFlag.AlignTop)

        self.guidance_card = QFrame()
        self.guidance_card.setObjectName("WakewordGuidanceCard")
        guidance_layout = QVBoxLayout(self.guidance_card)
        self._guidance_layout = guidance_layout
        guidance_layout.setContentsMargins(12, 10, 12, 10)
        guidance_layout.setSpacing(6)
        self.stage_badge_lbl = QLabel("Ready")
        self.stage_badge_lbl.setObjectName("WakewordStageBadge")
        self.stage_badge_lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.guidance_title_lbl = QLabel("Run a guided wakeword check")
        self.guidance_title_lbl.setObjectName("WakewordGuidanceTitle")
        self.guidance_body_lbl = QLabel(
            "You'll say the wakeword 5 times in the first part of the test, then you will read a short text for the second part of the test."
            "Follow the instructions which will appear below."
        )
        self.guidance_body_lbl.setObjectName("WakewordGuidanceBody")
        self.guidance_body_lbl.setWordWrap(True)
        self.guidance_hint_lbl = QLabel(
            "Adjust sensitivity first if needed. It will lock while the test is running."
        )
        self.guidance_hint_lbl.setObjectName("WakewordGuidanceHint")
        self.guidance_hint_lbl.setWordWrap(True)
        guidance_layout.addWidget(self.stage_badge_lbl, 0, Qt.AlignmentFlag.AlignLeft)
        guidance_layout.addWidget(self.guidance_title_lbl)
        guidance_layout.addWidget(self.guidance_body_lbl)
        guidance_layout.addWidget(self.guidance_hint_lbl)
        self.wakeword_selector = SelectorButton("Select Wakeword...", is_dark=self._is_dark)
        self.wakeword_selector.setMenu(QMenu(self.wakeword_selector))
        self.wakeword_selector.setFixedWidth(300)
        wakeword_row = QWidget()
        wakeword_row_layout = QHBoxLayout(wakeword_row)
        wakeword_row_layout.setContentsMargins(0, 0, 0, 0)
        wakeword_row_layout.setSpacing(8)
        wakeword_row_layout.addWidget(self.wakeword_selector)
        wakeword_row_layout.addStretch()
        guidance_layout.addWidget(wakeword_row)
        guidance_layout.addSpacing(4)
        self.primary_btn = QPushButton("Start Guided Test")
        self.primary_btn.setObjectName("WakewordPrimaryButton")
        guidance_layout.addWidget(self.primary_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self.alert_lbl = QLabel("")
        self.alert_lbl.setWordWrap(True)
        self.alert_lbl.setObjectName("WakewordAlertLabel")
        self.alert_lbl.setVisible(False)

        self.live_card = QFrame()
        self.live_card.setObjectName("WakewordLiveCard")
        live_layout = QVBoxLayout(self.live_card)
        live_layout.setContentsMargins(12, 10, 12, 10)
        live_layout.setSpacing(8)
        self.live_stage_badge_lbl = QLabel("Follow the instructions from this card now")
        self.live_stage_badge_lbl.setObjectName("WakewordStageBadge")
        self.live_stage_badge_lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.live_stage_badge_lbl.setVisible(False)
        self.live_instruction_bar = QProgressBar()
        self.live_instruction_bar.setObjectName("WakewordInstructionBar")
        self.live_instruction_bar.setRange(0, 100)
        self.live_instruction_bar.setValue(0)
        self.live_instruction_bar.setFormat("Say wakeword when prompted.")
        self.false_positive_text_lbl = QLabel("Read this text below at your normal pace and volume")
        self.false_positive_text_lbl.setObjectName("WakewordFalsePositivePromptLabel")
        self.false_positive_text_lbl.setVisible(False)
        self.false_positive_text_lbl.setWordWrap(True)
        self.false_positive_script_lbl = QLabel(
            '"Hey, have you ever noticed how certain phrases almost sound like something else when spoken aloud? I mean, when people say things like “hey, I know” or “they always,” the words can blur together in ways that might be picked up differently by a listening system."'
        )
        self.false_positive_script_lbl.setObjectName("WakewordFalsePositiveScriptLabel")
        self.false_positive_script_lbl.setVisible(False)
        self.false_positive_script_lbl.setWordWrap(True)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setFormat("Mic level %p%")
        self.attempt_progress = QProgressBar()
        self.attempt_progress.setObjectName("WakewordAttemptCounter")
        self.attempt_progress.setRange(0, self._test_state.attempt_target)
        self.attempt_progress.setValue(0)
        self.attempt_progress.setFormat("Attempts %v/%m")
        self.attempt_progress.setTextVisible(True)
        self.attempt_progress.setContentsMargins(0, 6, 0, 0)
        live_layout.addWidget(self.live_stage_badge_lbl, 0, Qt.AlignmentFlag.AlignLeft)
        live_layout.addWidget(self.live_instruction_bar)
        live_layout.addWidget(self.false_positive_text_lbl)
        live_layout.addWidget(self.false_positive_script_lbl)
        live_layout.addWidget(self.level_bar)
        live_layout.addWidget(self.attempt_progress)

        self.metrics_card = QFrame()
        self.metrics_card.setObjectName("WakewordResultsCard")
        metrics_layout = QVBoxLayout(self.metrics_card)
        metrics_layout.setContentsMargins(14, 12, 14, 12)
        metrics_layout.setSpacing(0)

        self.results_stage_badge_lbl = QLabel("Your Results")
        self.results_stage_badge_lbl.setObjectName("WakewordStageBadge")
        self.results_stage_badge_lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.results_stage_badge_lbl.setVisible(False)
        self.confidence_lbl = QLabel("Confidence: 0.00 | Smoothed: 0.00 (Low)")
        self.confidence_lbl.setVisible(False)

        self.verdict_lbl = QLabel("Not tested")
        self.verdict_lbl.setObjectName("WakewordResultsVerdict")
        self.verdict_lbl.setWordWrap(True)
        self.attempts_lbl = QLabel("Attempt progress: 0/5 complete")
        self.attempts_lbl.setObjectName("WakewordResultsMetric")
        self.false_pos_lbl = QLabel("False triggers: 0")
        self.false_pos_lbl.setObjectName("WakewordResultsMetric")
        self.confidence_stat_lbl = QLabel("")
        self.confidence_stat_lbl.setObjectName("WakewordResultsMetric")
        self.explanation_lbl = QLabel("")
        self.explanation_lbl.setObjectName("WakewordResultsDetail")
        self.explanation_lbl.setWordWrap(True)
        self.recommendation_lbl = QLabel("Recommended action: Run the test.")
        self.recommendation_lbl.setObjectName("WakewordResultsDetail")
        self.recommendation_lbl.setWordWrap(True)

        metrics_layout.addWidget(self.results_stage_badge_lbl, 0, Qt.AlignmentFlag.AlignLeft)
        metrics_layout.addSpacing(8)
        metrics_layout.addWidget(self.verdict_lbl)
        metrics_layout.addSpacing(10)
        metrics_layout.addWidget(self.attempts_lbl)
        metrics_layout.addSpacing(4)
        metrics_layout.addWidget(self.false_pos_lbl)
        metrics_layout.addSpacing(4)
        metrics_layout.addWidget(self.confidence_stat_lbl)
        metrics_layout.addSpacing(10)
        metrics_layout.addWidget(self.explanation_lbl)
        metrics_layout.addSpacing(4)
        metrics_layout.addWidget(self.recommendation_lbl)
        metrics_layout.addSpacing(10)

        results_btn_row = QHBoxLayout()
        results_btn_row.setContentsMargins(0, 0, 0, 0)
        results_btn_row.setSpacing(8)
        self.apply_btn = QPushButton("Apply wakeword")
        self.apply_btn.setObjectName("WakewordApplyButton")
        self.apply_btn.setProperty("class", "PrimaryActionButton BrandSuccessButton")
        self.apply_btn.setVisible(False)
        self.retest_btn = QPushButton("Retest")
        self.retest_btn.setObjectName("WakewordRetestButton")
        apply_brand_primary(self.retest_btn)
        self.retest_btn.setVisible(False)
        results_btn_row.addWidget(self.apply_btn)
        results_btn_row.addWidget(self.retest_btn)
        results_btn_row.addStretch()
        metrics_layout.addLayout(results_btn_row)

        self.advanced_card = QFrame()
        self.advanced_card.setObjectName("WakewordAdvancedCard")
        adv_layout = QVBoxLayout(self.advanced_card)
        adv_layout.setContentsMargins(12, 10, 12, 10)
        adv_layout.setSpacing(6)
        self.sensitivity_title_lbl = QLabel("Sensitivity tuning")
        self.sensitivity_title_lbl.setObjectName("WakewordAdvancedTitle")
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setMinimum(10)
        self.sensitivity_slider.setMaximum(95)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_value_lbl = QLabel("50%")
        self.sensitivity_value_lbl.setObjectName("WakewordSensitivityValueLabel")
        self.sensitivity_value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sensitivity_help_lbl = QLabel(
            "Higher sensitivity catches softer wakeword speech but may increase false triggers."
        )
        self.sensitivity_help_lbl.setWordWrap(True)
        self.sensitivity_lock_lbl = QLabel("")
        self.sensitivity_lock_lbl.setObjectName("WakewordAdvancedLockHint")
        self.sensitivity_lock_lbl.setWordWrap(True)
        self.sensitivity_delta_lbl = QLabel("")
        self.sensitivity_delta_lbl.setWordWrap(True)
        adv_layout.addWidget(self.sensitivity_title_lbl)
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(0, 0, 0, 0)
        slider_row.setSpacing(10)
        slider_row.addWidget(self.sensitivity_slider, 1)
        slider_row.addWidget(self.sensitivity_value_lbl, 0)
        adv_layout.addLayout(slider_row)
        adv_layout.addWidget(self.sensitivity_help_lbl)
        adv_layout.addWidget(self.sensitivity_lock_lbl)
        adv_layout.addWidget(self.sensitivity_delta_lbl)

        root.addLayout(header_row)
        root.addWidget(self.guidance_card)
        root.addWidget(self.live_card)
        root.addWidget(self.metrics_card)
        root.addWidget(self.advanced_card)
        root.addStretch(1)
        outer.addWidget(self.container)

    def _apply_theme_styles(self) -> None:
        if self._is_dark:
            bg = "#1e1e2e"
            fg = "#cdd6f4"
            card = "#232337"
            border = "rgba(255, 255, 255, 0.10)"
            subtext = "rgba(205, 214, 244, 0.75)"
            alert = "#f59e0b"
            header_subtext = "rgba(205, 214, 244, 0.72)"
        else:
            bg = "#f8fafc"
            fg = "#1e293b"
            card = "#ffffff"
            border = "#cbd5e1"
            subtext = "#64748b"
            alert = "#ea580c"
            header_subtext = "#64748b"

        self.container.setStyleSheet(
            f"""
            QFrame#WakewordLabContainer {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 14px;
            }}
            QFrame#WakewordGuidanceCard, QFrame#WakewordLiveCard, QFrame#WakewordResultsCard, QFrame#WakewordAdvancedCard {{
                background-color: {card};
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QFrame#WakewordGuidanceCard[state="attention"] {{ border: 1px solid #3b82f6; }}
            QFrame#WakewordGuidanceCard[state="cancelled"] {{ border: 1px solid #dc2626; }}
            QFrame#WakewordLiveCard[state="attention"] {{ border: 1px solid #3b82f6; }}
            QFrame#WakewordResultsCard[state="success"] {{ border: 1px solid #16a34a; }}
            QFrame#WakewordResultsCard[state="caution"] {{ border: 1px solid #eab308; }}
            QFrame#WakewordResultsCard[state="failure"] {{ border: 1px solid #dc2626; }}
            QLabel {{ color: {fg}; }}
            QLabel#WakewordHeaderTitle {{ font-size: 18px; font-weight: 700; }}
            QLabel#WakewordHeaderSubtitle {{ color: {header_subtext}; font-size: 12px; }}
            QLabel#WakewordStageBadge {{
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border-radius: 8px;
                background: rgba(139, 92, 246, 0.16);
                color: #a78bfa;
            }}
            QLabel#WakewordStageBadge[state="attention"] {{ background: rgba(59, 130, 246, 0.16); color: #3b82f6; }}
            QLabel#WakewordStageBadge[state="false_positive"] {{ background: rgba(245, 158, 11, 0.16); color: #f59e0b; }}
            QLabel#WakewordStageBadge[state="success"] {{ background: rgba(34, 197, 94, 0.16); color: #16a34a; }}
            QLabel#WakewordStageBadge[state="warning"] {{ background: rgba(234, 88, 12, 0.16); color: #ea580c; }}
            QLabel#WakewordStageBadge[state="cancelled"] {{ background: rgba(239, 68, 68, 0.16); color: #dc2626; }}
            QLabel#WakewordStageBadge[state="error"] {{ background: rgba(239, 68, 68, 0.16); color: #dc2626; }}
            QLabel#WakewordGuidanceTitle {{ font-size: 15px; font-weight: 600; }}
            QLabel#WakewordGuidanceHint, QLabel#WakewordAdvancedLockHint {{ color: {subtext}; font-size: 12px; }}
            QLabel#WakewordAlertLabel {{ color: {alert}; font-weight: 600; }}
            QLabel#WakewordAdvancedTitle {{ color: {fg}; font-size: 12px; font-weight: 700; }}
            QLabel#WakewordResultsVerdict {{ font-size: 15px; font-weight: 700; }}
            QLabel#WakewordResultsVerdict[result_tone="success"] {{ color: #16a34a; }}
            QLabel#WakewordResultsVerdict[result_tone="caution"] {{ color: #eab308; }}
            QLabel#WakewordResultsVerdict[result_tone="failure"] {{ color: #dc2626; }}
            QLabel#WakewordResultsMetric {{ font-size: 13px; }}
            QLabel#WakewordResultsDetail {{ color: {subtext}; font-size: 12px; }}
            QLabel#WakewordFalsePositivePromptLabel {{
                color: #3b82f6;
                font-size: 12px;
                font-weight: 700;
            }}
            QLabel#WakewordFalsePositiveScriptLabel {{
                color: {fg};
                font-size: 12px;
                font-style: italic;
            }}
            QPushButton#WakewordHeaderCloseButton {{
                border: 1px solid {border};
                border-radius: 8px;
                padding: 4px;
                font-weight: 700;
            }}
            QPushButton#WakewordApplyButton[result_tone="success"] {{
                background-color: #16a34a;
                color: #ffffff;
                border: 1px solid #16a34a;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }}
            QPushButton#WakewordApplyButton[result_tone="caution"] {{
                background-color: #eab308;
                color: #1e293b;
                border: 1px solid #eab308;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }}
            QPushButton#WakewordApplyButton[result_tone="failure"] {{
                background-color: #dc2626;
                color: #ffffff;
                border: 1px solid #dc2626;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }}
            QProgressBar {{
                background-color: transparent;
                border: 1px solid {border};
                border-radius: 6px;
                text-align: center;
                color: {subtext};
                min-height: 16px;
            }}
            QProgressBar::chunk {{
                background-color: #8b5cf6;
                border-radius: 5px;
            }}
            QProgressBar#WakewordInstructionBar {{
                background-color: transparent;
                border: 1px solid transparent;
                padding: 2px 0px;
                font-weight: 700;
                color: #3b82f6;
            }}
            QProgressBar#WakewordInstructionBar::chunk {{
                background-color: transparent;
                border-radius: 0px;
            }}
            QProgressBar#WakewordInstructionBar[state="countdown"] {{ color: #ea580c; }}
            QProgressBar#WakewordInstructionBar[state="listening"] {{ color: #16a34a; }}
            QProgressBar#WakewordAttemptCounter {{
                background-color: transparent;
                border: 1px solid transparent;
                padding: 4px 0px 0px 0px;
            }}
            QProgressBar#WakewordAttemptCounter::chunk {{
                background-color: transparent;
                border-radius: 0px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: transparent;
                border: 1px solid {border};
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: #8b5cf6;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
                background: #8b5cf6;
                border: 1px solid {border};
            }}
            """
        )
        self._refresh_widget_style(self.guidance_card)
        self._refresh_widget_style(self.stage_badge_lbl)

    def refresh_theme(self, is_dark: bool) -> None:
        """Re-apply all dialog styles after an app-wide theme toggle."""
        self._is_dark = is_dark
        self._apply_theme_styles()
        menu = self.wakeword_selector.menu()
        if menu:
            self._apply_menu_theme(menu, is_dark)
        self._apply_settings_menu_button_chevron_state(self.wakeword_selector)

    def _bind_signals(self) -> None:
        self.primary_btn.clicked.connect(self._on_primary_action)
        self.apply_btn.clicked.connect(self._apply_result)
        self.retest_btn.clicked.connect(self._start_test)
        self.header_close_btn.clicked.connect(lambda: self._cancel_test(close_after=True))
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        if self.audio_worker:
            self.audio_worker.volume_update.connect(self._on_volume_update)
            self.audio_worker.wakeword_score_update.connect(self._on_confidence_update)
            self.audio_worker.wakeword_test_detection.connect(self._on_test_detection)
            self.audio_worker.wakeword_model_error.connect(self._on_model_error)

    def _set_optional_label_text(self, label: QLabel, text: str) -> None:
        text_s = str(text or "")
        label.setText(text_s)
        label.setVisible(bool(text_s.strip()))
        label.updateGeometry()

    def _refresh_dialog_layouts(self) -> None:
        for widget in (
            self.header_title_lbl,
            self.header_subtitle_lbl,
            self.stage_badge_lbl,
            self.guidance_title_lbl,
            self.guidance_body_lbl,
            self.guidance_hint_lbl,
            self.wakeword_selector,
            self.primary_btn,
            self.live_card,
            self.metrics_card,
            self.advanced_card,
        ):
            widget.updateGeometry()
        if hasattr(self, "_guidance_layout"):
            self._guidance_layout.invalidate()
        if hasattr(self, "_root_layout"):
            self._root_layout.invalidate()
            self._root_layout.activate()
        self.container.updateGeometry()
        self.updateGeometry()

    def on_wakeword_selection_changed(self, sync_catalog: bool = True) -> None:
        self._test_attempt_open = False
        self._test_attempt_mode = "idle"
        self._test_attempt_timer.stop()
        self._test_false_positive_timer.stop()
        self._test_phase = "ready"
        if self.audio_worker:
            self.audio_worker.set_test_mode(False)
        self._has_fresh_completed_test = False
        self._last_test_summary = None
        self._sync_sensitivity_from_worker()
        if sync_catalog:
            self._sync_wakeword_catalog()
        self.attempt_progress.setValue(0)
        self.attempts_lbl.setText("Attempt progress: 0/5 complete")
        self.false_pos_lbl.setText("False triggers: 0")
        self.confidence_stat_lbl.setText("")
        self.verdict_lbl.setText("Not tested")
        self.explanation_lbl.setText("")
        self.recommendation_lbl.setText("Recommended action: Run the test.")
        self._set_view_state("ready")

    def _sync_wakeword_catalog(self) -> None:
        if not self.audio_worker:
            return
        recommended = [
            ("Recommended - " + spec.display_name, spec.display_name)
            for spec in self.audio_worker.wakeword_manager.list_recommended()
        ]
        community = [
            ("Community - " + spec.display_name, spec.display_name)
            for spec in self.audio_worker.wakeword_manager.list_community()
        ]
        wakeword_items = recommended + community
        if not wakeword_items:
            return
        self._build_prestige_menu(
            self.wakeword_selector,
            wakeword_items,
            self._on_wakeword_selector_changed,
        )
        active_name = getattr(self.audio_worker, "active_wakeword_name", "") or wakeword_items[0][1]
        matching_label = next(
            (label for label, data in wakeword_items if data == active_name),
            wakeword_items[0][0],
        )
        self.wakeword_selector.setText(matching_label)
        self._apply_settings_menu_button_chevron_state(self.wakeword_selector)

    def _build_prestige_menu(self, button, items, callback) -> None:
        menu = QMenu(button)
        menu.setObjectName("PrestigeMenu")
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._apply_menu_theme(menu, self._is_dark)

        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for label, _data in items:
            list_widget.addItem(label)

        required_height = len(items) * 32 + 10
        max_height = int(self.height() * 0.5) if self.height() else 400
        list_widget.setFixedHeight(min(required_height, max_height))

        def sync_dropdown_width():
            content_w = list_widget.sizeHintForColumn(0) + 40
            list_widget.setFixedWidth(max(button.width() - 8, content_w, 220))

        menu.aboutToShow.connect(sync_dropdown_width)

        def on_item_clicked(item):
            selected_label = item.text()
            matched_data = next((d for l, d in items if l == selected_label), selected_label)
            button.setText(selected_label)
            callback(matched_data)
            menu.hide()

        list_widget.itemClicked.connect(on_item_clicked)

        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)
        button.setMenu(menu)

    def _apply_menu_theme(self, menu, is_dark: bool) -> None:
        from PyQt6.QtGui import QColor, QPalette

        palette = QPalette()
        if is_dark:
            bg = QColor("#1e1e2e")
            fg = QColor("#cdd6f4")
            sel_fg = QColor("#cdd6f4")
            border = "rgba(255, 255, 255, 0.1)"
            hover = "#313244"
        else:
            bg = QColor("#ffffff")
            fg = QColor("#1e293b")
            sel_fg = QColor("#0f172a")
            border = "#cbd5e1"
            hover = "#f1f5f9"

        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(hover))
        palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)

        menu.setPalette(palette)
        menu.setStyleSheet(
            f"""
            QMenu {{ background-color: {bg.name()}; border: 1px solid {border}; border-radius: 6px; padding: 4px; }}
            QListWidget#PrestigeMenuList {{ background-color: transparent; border: none; outline: none; }}
            QListWidget#PrestigeMenuList::item {{ background-color: transparent; color: {fg.name()}; padding: 8px 25px; border-radius: 4px; min-height: 24px; }}
            QListWidget#PrestigeMenuList::item:selected, QListWidget#PrestigeMenuList::item:hover {{ background-color: {hover}; color: {sel_fg.name()}; }}
            QScrollBar:vertical {{ border: none; background: transparent; width: 6px; margin: 0px; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 3px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            """
        )

    def _apply_settings_menu_button_chevron_state(self, button: QPushButton) -> None:
        if isinstance(button, SelectorButton):
            button.apply_theme(self._is_dark)
            return
        muted = "#3f3f46" if self._is_dark else "#a1a1aa"
        active = "#64748b"
        button.setIcon(qta.icon("fa5s.chevron-down", color=active if button.isEnabled() else muted))

    def _on_wakeword_selector_changed(self, display_name: str) -> None:
        if not self.audio_worker:
            return
        self.audio_worker.set_wakeword(display_name)
        self.on_wakeword_selection_changed(sync_catalog=False)

    def _sync_sensitivity_from_worker(self) -> None:
        threshold = float(getattr(self.audio_worker, "active_wakeword_threshold", 0.5)) if self.audio_worker else 0.5
        value = max(10, min(95, int((1.0 - threshold) * 100)))
        self.sensitivity_slider.blockSignals(True)
        self.sensitivity_slider.setValue(value)
        self.sensitivity_slider.blockSignals(False)
        self.sensitivity_value_lbl.setText(f"{value}%")

    def _on_volume_update(self, level: float) -> None:
        self.level_bar.setValue(max(0, min(100, int(level * 100))))

    def _on_confidence_update(self, raw: float, smoothed: float) -> None:
        self.confidence_lbl.setText(
            f"Confidence: {raw:.2f} | Smoothed: {smoothed:.2f} ({self._confidence_band(smoothed)})"
        )
        if self._test_phase == "attempts":
            self._test_attempt_peak = max(self._test_attempt_peak, float(smoothed))

    def _start_test(self) -> None:
        self._test_state = WakewordTestbedState()
        self._test_attempt_index = 0
        self._test_attempt_peak = 0.0
        self._test_phase = "attempts"
        self._test_attempt_open = False
        self._test_attempt_mode = "idle"
        self._has_fresh_completed_test = False
        self._last_test_summary = None
        self.attempt_progress.setRange(0, self._test_state.attempt_target)
        self.attempt_progress.setValue(0)
        self.attempts_lbl.setText("Attempt progress: 0/5 complete")
        self.false_pos_lbl.setText("False triggers: 0")
        self.confidence_stat_lbl.setText("")
        self.verdict_lbl.setText("Testing in progress...")
        self.explanation_lbl.setText("")
        self.recommendation_lbl.setText("Recommended action: Complete the full test cycle.")
        self.sensitivity_delta_lbl.setText("")
        self.apply_btn.setVisible(False)
        self._set_live_instruction(
            "Say wakeword when prompted. Each attempt includes a short prep countdown.",
            total=100,
            value=0,
        )
        if self.audio_worker:
            self.audio_worker.set_test_mode(True)
        self._set_view_state("attempts")
        self._start_next_attempt()

    def _start_next_attempt(self) -> None:
        self._test_attempt_peak = 0.0
        self._test_attempt_open = False
        self._test_attempt_mode = "countdown"
        self._test_attempt_seconds_left = self._test_state.attempt_prep_seconds
        self._update_attempt_instruction()
        self._test_attempt_timer.start()

    def _update_attempt_instruction(self) -> None:
        attempt_num = self._test_attempt_index + 1
        total = self._test_state.attempt_target
        wakeword = self._current_wakeword_label()
        prep_total = max(1, self._test_state.attempt_prep_seconds)
        listen_total = max(1, self._test_attempt_timeout_secs)
        if self._test_attempt_mode == "countdown":
            left = max(0, self._test_attempt_seconds_left)
            if left > 0:
                self.guidance_body_lbl.setText(
                    f"Attempt {attempt_num} of {total}. "
                    f"Say wakeword in {left}..."
                )
                self._set_live_instruction(
                    self._countdown_prompt(wakeword, left),
                    total=prep_total,
                    value=prep_total - left,
                )
                return
            self.guidance_body_lbl.setText(
                f"Attempt {attempt_num} of {total}. Go - say wakeword now."
            )
            self._set_live_instruction(
                f"Say {wakeword} now!",
                total=prep_total,
                value=prep_total,
            )
            return
        if self._test_attempt_mode == "listening":
            left = max(0, self._test_attempt_seconds_left)
            self.guidance_body_lbl.setText(
                f"Attempt {attempt_num} of {total}. "
                f"Listening now ({left}s left)."
            )
            self._set_live_instruction(
                f"Listening for {wakeword}...",
                total=listen_total,
                value=listen_total - left,
            )
            return
        self.guidance_body_lbl.setText(
            f"Attempt {attempt_num} of {total}. Waiting for next attempt."
        )
        self._set_live_instruction(
            f"Attempt {attempt_num}/{total} - Waiting",
            total=100,
            value=0,
        )

    def _countdown_prompt(self, wakeword: str, seconds_left: int) -> str:
        left = max(1, int(seconds_left))
        if left == 3:
            return f'Say "{wakeword}" in 3 .... 2 .... 1.... Go!'
        if left == 2:
            return f'Say "{wakeword}" in 2 .... 1.... Go!'
        return f'Say "{wakeword}" in 1.... Go!'

    def _current_wakeword_label(self) -> str:
        if not self.audio_worker:
            return "wakeword"
        name = str(getattr(self.audio_worker, "active_wakeword_name", "") or "").strip()
        return name if name else "wakeword"

    def _on_attempt_tick(self) -> None:
        if self._test_phase != "attempts" or not self._test_attempt_open:
            if self._test_attempt_mode != "countdown":
                self._test_attempt_timer.stop()
                return
        self._test_attempt_seconds_left -= 1
        if self._test_attempt_mode == "countdown":
            if self._test_attempt_seconds_left > 0:
                self._update_attempt_instruction()
                return
            self._test_attempt_mode = "listening"
            self._test_attempt_open = True
            self._test_attempt_seconds_left = self._test_attempt_timeout_secs
            self._update_attempt_instruction()
            return
        if self._test_attempt_seconds_left > 0:
            self._update_attempt_instruction()
            return
        self._test_attempt_timer.stop()
        self._test_attempt_open = False
        self._test_attempt_mode = "idle"
        self._test_state.record_detection_attempt(
            peak_confidence=max(self._test_attempt_peak, 0.0),
            detected=False,
            timed_out=True,
        )
        self._test_attempt_index += 1
        self.attempt_progress.setValue(self._test_attempt_index)
        self.attempts_lbl.setText(f"Attempt {self._test_attempt_index}/{self._test_state.attempt_target}: missed")
        if self._test_attempt_index >= self._test_state.attempt_target:
            self._await_false_positive_start()
        else:
            self._start_next_attempt()

    def _on_test_detection(self, score: float) -> None:
        if self._test_phase == "attempts" and self._test_attempt_open:
            self._test_attempt_open = False
            self._test_attempt_mode = "idle"
            self._test_attempt_timer.stop()
            self._test_state.record_detection_attempt(
                peak_confidence=max(self._test_attempt_peak, float(score)),
                detected=True,
                timed_out=False,
            )
            self._test_attempt_index += 1
            self._test_attempt_peak = 0.0
            self.attempt_progress.setValue(self._test_attempt_index)
            self.attempts_lbl.setText(f"Attempt {self._test_attempt_index}/{self._test_state.attempt_target}: detected")
            if self._test_attempt_index >= self._test_state.attempt_target:
                self._await_false_positive_start()
            else:
                self._start_next_attempt()
        elif self._test_phase == "false_positive":
            self._test_state.record_false_trigger()
            self.false_pos_lbl.setText(f"False triggers: {self._test_state.false_triggers}")

    def _await_false_positive_start(self) -> None:
        self._test_phase = "awaiting_fp_start"
        self._set_live_instruction(
            "Attempts complete. Click Begin normal speech check.",
            total=100,
            value=100,
        )
        self._set_view_state("awaiting_fp_start")

    def _start_false_positive_phase(self) -> None:
        self._test_phase = "false_positive"
        self._false_positive_seconds_left = self._test_state.false_positive_seconds
        self._set_live_instruction(
            "",
            total=self._test_state.false_positive_seconds,
            value=0,
        )
        self._set_view_state("false_positive")
        self._test_false_positive_timer.start()

    def _on_false_positive_tick(self) -> None:
        self._false_positive_seconds_left -= 1
        total = self._test_state.false_positive_seconds
        self._set_live_instruction(
            "",
            total=total,
            value=0,
        )
        if self._false_positive_seconds_left <= 0:
            self._test_false_positive_timer.stop()
            self._complete_test()

    def _complete_test(self) -> None:
        self._test_phase = "done"
        self._test_attempt_open = False
        self._test_attempt_mode = "idle"
        self._test_attempt_timer.stop()
        if self.audio_worker:
            self.audio_worker.set_test_mode(False)
        summary = self._test_state.build_summary()
        self._last_test_summary = summary
        self._has_fresh_completed_test = True
        self.attempts_lbl.setText(
            f"Detected {summary.successes}/{self._test_state.attempt_target} (missed {summary.missed_attempts})"
        )
        self.false_pos_lbl.setText(f"False triggers: {summary.false_triggers}")
        self.confidence_stat_lbl.setText(
            f"Avg confidence: {summary.average_confidence:.2f}  \u00b7  Consistency: {summary.consistency_label}"
        )
        self.verdict_lbl.setText(summary.reliability_label)
        self.explanation_lbl.setText(summary.explanation)
        self.recommendation_lbl.setText(f"Recommended action: {summary.recommendation}")
        self._set_live_instruction(
            "Test complete. Review reliability and apply or retest.",
            total=100,
            value=100,
        )
        self._auto_calibrate_threshold(summary)
        result_tone = self._result_tone(summary)
        if result_tone == "success":
            self._set_view_state("success")
        elif result_tone == "caution":
            self._set_view_state("caution")
        else:
            self._set_view_state("failure")

    def _auto_calibrate_threshold(self, summary) -> None:
        if not self.audio_worker:
            return
        current = float(getattr(self.audio_worker, "active_wakeword_threshold", 0.5))
        adjusted = current
        if summary.false_triggers > 0:
            adjusted = min(0.9, current + 0.05)
        elif summary.successes < self._test_state.attempt_target:
            adjusted = max(0.2, current - 0.03)
        self.audio_worker.set_wakeword_threshold(adjusted)
        before = max(10, min(95, int((1.0 - current) * 100)))
        after = max(10, min(95, int((1.0 - adjusted) * 100)))
        self.sensitivity_slider.blockSignals(True)
        self.sensitivity_slider.setValue(after)
        self.sensitivity_slider.blockSignals(False)
        self.sensitivity_value_lbl.setText(f"{after}%")
        self.sensitivity_delta_lbl.setText(
            f"Sensitivity automatically adjusted from {before}% to {after}% based on test results."
            if before != after
            else ""
        )

    def _on_sensitivity_changed(self, value: int) -> None:
        self.sensitivity_value_lbl.setText(f"{int(value)}%")
        if not self.audio_worker:
            return
        threshold = max(0.1, min(0.95, 1.0 - (float(value) / 100.0)))
        self.audio_worker.set_wakeword_threshold(threshold)

    def _on_model_error(self, message: str) -> None:
        self.guidance_body_lbl.setText(f"Wakeword model error: {message}")
        self._set_view_state("error")

    def _apply_result(self) -> None:
        if not self._has_fresh_completed_test:
            self.guidance_body_lbl.setText("Run a complete test before applying this wakeword.")
            return
        summary = self._last_test_summary
        if summary and summary.reliability_label == "Not recommended":
            is_dark = getattr(self.parent(), "_is_dark_theme", True)
            warn = PrestigeDialog(
                self.parent(),
                "Apply not-recommended wakeword?",
                "This wakeword scored as not recommended in your environment. Apply anyway?",
                is_dark=is_dark,
            )
            if not warn.exec():
                return
        self.guidance_body_lbl.setText("Wakeword applied. Detection will use this validated configuration.")
        self.accept()

    @staticmethod
    def _result_tone(summary) -> str:
        if not summary:
            return "default"
        if summary.reliability_label == "Works great":
            return "success"
        if summary.reliability_label == "Usable with minor issues":
            return "caution"
        return "failure"

    def _on_primary_action(self) -> None:
        if self._view_state in {"attempts", "false_positive"}:
            self._cancel_test(close_after=False)
            return
        if self._view_state == "awaiting_fp_start":
            self._start_false_positive_phase()
            return
        if self._view_state in {"ready", "cancelled", "error"}:
            self._start_test()
            return
        if self._view_state in {"success", "caution", "failure"}:
            self._apply_result()

    def _cancel_test(self, close_after: bool = False) -> None:
        was_running = self._view_state in {"attempts", "false_positive"}
        self._test_attempt_open = False
        self._test_attempt_mode = "idle"
        self._test_attempt_timer.stop()
        self._test_false_positive_timer.stop()
        if self.audio_worker:
            self.audio_worker.set_test_mode(False)
        if close_after:
            self.close()
            return
        if was_running:
            self._test_phase = "cancelled"
            self._set_view_state("cancelled")

    def _set_view_state(self, state: str) -> None:
        self._view_state = str(state or "ready")
        guidance_card_state = "default"
        live_card_state = "default"
        metrics_card_state = "default"
        top_badge_visible = False
        top_badge_text = "Ready"
        top_badge_state = "default"
        live_badge_visible = False
        live_badge_state = "attention"
        results_badge_visible = False
        results_badge_text = "Your Results"
        results_badge_state = "attention"
        if self._view_state == "ready":
            top_badge_visible = True
            top_badge_text = "Ready"
            self.guidance_title_lbl.setText("Run a guided wakeword check")
            self.guidance_body_lbl.setText(
                "You'll be asked to say the wakeword 5 times in a row for the first part of the test.\n"
                "For the second part of the test, you will read a small text that will be provided below."
            )
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._set_live_instruction(
                "Say wakeword when prompted. Countdown appears before each attempt.",
                total=100,
                value=0,
            )
        elif self._view_state == "attempts":
            live_card_state = "attention"
            live_badge_visible = True
            self.guidance_title_lbl.setText("Testing Wakeword Detection")
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._update_attempt_instruction()
        elif self._view_state == "awaiting_fp_start":
            guidance_card_state = "attention"
            top_badge_visible = True
            top_badge_text = "Ready for Part 2"
            top_badge_state = "attention"
            self.guidance_title_lbl.setText("Testing Wakeword False-Positives")
            self.guidance_body_lbl.setText(
                "Wakeword detection test complete. Click Begin normal speech check to test for false-positives."
            )
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._set_live_instruction(
                "Wakeword detection test complete. Click Begin normal speech check to test for false-positives.",
                total=100,
                value=100,
            )
        elif self._view_state == "false_positive":
            live_card_state = "attention"
            live_badge_visible = True
            self.guidance_title_lbl.setText("Speak normally")
            self.guidance_body_lbl.setText("Read the text below at your normal volume and speed, and wait for the results to appear.")
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._set_live_instruction(
                "",
                total=max(1, self._test_state.false_positive_seconds),
                value=0,
            )
        elif self._view_state == "success":
            metrics_card_state = "success"
            results_badge_visible = True
            results_badge_text = "Test Passed"
            results_badge_state = "success"
            self.verdict_lbl.setProperty("result_tone", "success")
            self.guidance_title_lbl.setText("This wakeword looks reliable")
            self.guidance_body_lbl.setText(
                "Detection stayed stable and false-trigger risk remained low."
            )
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._set_live_instruction(
                "Test complete - reliability looks strong.",
                total=100,
                value=100,
            )
        elif self._view_state == "caution":
            metrics_card_state = "caution"
            results_badge_visible = True
            results_badge_text = "Needs Review"
            results_badge_state = "warning"
            self.verdict_lbl.setProperty("result_tone", "caution")
            self.guidance_title_lbl.setText("This wakeword needs caution")
            self.guidance_body_lbl.setText(
                "Qube saw missed detections or false triggers in your environment."
            )
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._set_live_instruction(
                "Test complete - retest recommended before applying.",
                total=100,
                value=100,
            )
        elif self._view_state == "failure":
            metrics_card_state = "failure"
            results_badge_visible = True
            results_badge_text = "Not Recommended"
            results_badge_state = "error"
            self.verdict_lbl.setProperty("result_tone", "failure")
            self.guidance_title_lbl.setText("This wakeword is not recommended")
            self.guidance_body_lbl.setText(
                "Qube saw reliability problems that make this wakeword risky in your environment."
            )
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._set_live_instruction(
                "Test complete - applying this wakeword carries significant risk.",
                total=100,
                value=100,
            )
        elif self._view_state == "cancelled":
            guidance_card_state = "cancelled"
            top_badge_visible = True
            top_badge_text = "Cancelled"
            top_badge_state = "cancelled"
            self.guidance_title_lbl.setText("Test stopped")
            self.guidance_body_lbl.setText("The guided run was cancelled before completion.")
            self._set_optional_label_text(
                self.guidance_hint_lbl,
                "Select your wakeword and start a new test when you're ready.",
            )
            self._set_live_instruction(
                "Test cancelled. Start again when ready.",
                total=100,
                value=0,
            )
        elif self._view_state == "error":
            top_badge_visible = True
            top_badge_text = "Error"
            top_badge_state = "error"
            self.guidance_title_lbl.setText("Wakeword test unavailable")
            self._set_optional_label_text(self.guidance_hint_lbl, "")
            self._set_live_instruction(
                "Wakeword test unavailable. Retry or switch wakeword.",
                total=100,
                value=0,
            )

        self.stage_badge_lbl.setText(top_badge_text)
        self.stage_badge_lbl.setVisible(top_badge_visible)
        self.stage_badge_lbl.setProperty("state", top_badge_state)
        self.live_stage_badge_lbl.setVisible(live_badge_visible)
        self.live_stage_badge_lbl.setProperty("state", live_badge_state)
        self.results_stage_badge_lbl.setText(results_badge_text)
        self.results_stage_badge_lbl.setVisible(results_badge_visible)
        self.results_stage_badge_lbl.setProperty("state", results_badge_state)
        self.guidance_card.setProperty("state", guidance_card_state)
        self.live_card.setProperty("state", live_card_state)
        self.metrics_card.setProperty("state", metrics_card_state)
        running = self._view_state in {"attempts", "false_positive"}
        selector_locked = self._view_state in {"attempts", "awaiting_fp_start", "false_positive"}
        self.wakeword_selector.setEnabled(not selector_locked)
        self._apply_settings_menu_button_chevron_state(self.wakeword_selector)
        self.sensitivity_slider.setEnabled(not running)
        self.attempt_progress.setVisible(self._view_state == "attempts")
        self.apply_btn.setVisible(self._view_state in {"success", "caution", "failure"})
        self.retest_btn.setVisible(self._view_state in {"success", "caution", "failure"})
        self.metrics_card.setVisible(self._view_state in {"success", "caution", "failure"})
        in_false_positive = self._view_state == "false_positive"
        self.live_instruction_bar.setVisible(not in_false_positive)
        self.false_positive_text_lbl.setVisible(in_false_positive)
        self.false_positive_script_lbl.setVisible(in_false_positive)
        self.sensitivity_lock_lbl.setText(
            "Sensitivity is locked while testing runs."
            if running
            else "Sensitivity is available before and after test runs."
        )
        self._sync_action_buttons()
        self._refresh_widget_style(self.guidance_card)
        self._refresh_widget_style(self.live_card)
        self._refresh_widget_style(self.metrics_card)
        self._refresh_widget_style(self.stage_badge_lbl)
        self._refresh_widget_style(self.live_stage_badge_lbl)
        self._refresh_widget_style(self.results_stage_badge_lbl)
        self._refresh_widget_style(self.verdict_lbl)
        self._refresh_dialog_layouts()

    def _sync_action_buttons(self) -> None:
        # `apply_btn` intentionally retains container-level QSS driven by its
        # `result_tone` dynamic property (see self.container.setStyleSheet
        # `#WakewordApplyButton[result_tone="..."]` rules). Do NOT route it
        # through the brand_buttons helper — a widget-level stylesheet would
        # defeat the container's tone-based variants.
        if self._view_state in {"attempts", "false_positive"}:
            self.primary_btn.setText("Cancel test")
            apply_brand_danger(self.primary_btn)
            self.primary_btn.setVisible(True)
        elif self._view_state == "awaiting_fp_start":
            self.primary_btn.setText("Begin normal speech check")
            apply_brand_primary(self.primary_btn)
            self.primary_btn.setVisible(True)
        elif self._view_state in {"success", "caution", "failure"}:
            self.primary_btn.setVisible(False)
            self.apply_btn.setText("Apply wakeword" if self._view_state == "success" else "Apply anyway")
            self.apply_btn.setProperty("class", "")
            result_tone = "success" if self._view_state == "success" else self._view_state
            self.apply_btn.setProperty("result_tone", result_tone)
            self._refresh_widget_style(self.apply_btn)
            apply_brand_primary(self.retest_btn)
            self.retest_btn.setProperty("result_tone", "default")
            return
        elif self._view_state == "error":
            self.primary_btn.setText("Retry test")
            apply_brand_primary(self.primary_btn)
            self.primary_btn.setVisible(True)
        else:
            self.primary_btn.setText("Start Guided Test")
            apply_brand_primary(self.primary_btn)
            self.primary_btn.setVisible(True)
            self.apply_btn.setProperty("result_tone", "default")
            self.retest_btn.setProperty("result_tone", "default")

    @staticmethod
    def _refresh_widget_style(widget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _set_live_instruction(self, text: str, total: int, value: int) -> None:
        total_i = max(1, int(total))
        value_i = max(0, min(int(value), total_i))
        self.live_instruction_bar.setRange(0, total_i)
        self.live_instruction_bar.setValue(value_i)
        text_s = str(text)
        self.live_instruction_bar.setFormat(text_s)
        tone = "info"
        if text_s.startswith("Listening for"):
            tone = "listening"
        elif "when prompted" not in text_s and (text_s.startswith('Say "') or text_s.startswith("Say ")):
            tone = "countdown"
        self.live_instruction_bar.setProperty("state", tone)
        self._refresh_widget_style(self.live_instruction_bar)

    def _confidence_band(self, smoothed: float) -> str:
        if smoothed >= 0.75:
            return "Strong"
        if smoothed >= 0.4:
            return "Moderate"
        return "Low"

    def closeEvent(self, event) -> None:
        self._test_attempt_open = False
        self._test_attempt_timer.stop()
        self._test_false_positive_timer.stop()
        if self.audio_worker:
            self.audio_worker.set_test_mode(False)
        super().closeEvent(event)
