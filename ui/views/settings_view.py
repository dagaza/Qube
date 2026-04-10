import os
import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QPushButton, QFileDialog,
    QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QProgressBar,
    QStyledItemDelegate, QListView, QMenu, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from pathlib import Path
import logging

from core.audio_utils import get_input_devices, get_output_devices
from core.network import is_port_open

logger = logging.getLogger("Qube.UI.Settings")
class SettingsView(QWidget):
    audio_pin_toggle = pyqtSignal(bool)
    auto_activator_toggle = pyqtSignal(bool) # 🔑 ADD THIS
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        # Reference our workers
        self.downloader = workers.get("downloader")
        self.audio_worker = workers.get("audio")
        self.tts_worker = workers.get("tts")
        self.llm_worker = workers.get("llm")

        # Track UI elements for models
        self.model_cards = {}

        self._setup_ui()
        self._wire_downloader()
        self._populate_hardware_selectors()
    
    def _wire_downloader(self):
        """Connects the background download thread to the UI progress bars."""
        if self.downloader:
            self.downloader.progress_update.connect(self._update_download_ui)
            self.downloader.download_complete.connect(self._on_download_finished)

    def _build_tts_hub(self) -> QWidget:
        """The 'App Store' style grid for TTS models."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # 1. 🔑 MODEL METADATA (The 'Store' catalog)
        models = [
            {
                "id": "kokoro",
                "name": "Kokoro (Featherweight)",
                "desc": "Ultra-fast, tiny footprint. Perfect for standard interactions.",
                "size": "80 MB",
                "path": "models/tts/kokoro/kokoro-v1.0.onnx"
            },
            {
                "id": "f5",
                "name": "F5-TTS (Middleweight)",
                "desc": "Premium zero-shot voice cloning. Great balance of quality and speed.",
                "size": "1.2 GB",
                "path": "models/tts/f5/model.safetensors"
            },
            {
                "id": "voxtral",
                "name": "Voxtral (Heavyweight)",
                "desc": "State-of-the-Art emotional intelligence. March 2026 flagship model.",
                "size": "3.5 GB",
                "path": "models/tts/voxtral/model.safetensors"
            }
        ]

        for m in models:
            card = self._create_model_card(m)
            self.model_cards[m["id"]] = card
            layout.addWidget(card)
            
        # 2. 🔑 THE HACKER API BOX
        api_container = QWidget()
        api_layout = QHBoxLayout(api_container)
        self.custom_api_input = QLineEdit()
        self.custom_api_input.setPlaceholderText("Custom OpenAI TTS URL (e.g. http://localhost:5050/v1)")
        api_btn = QPushButton("Connect")
        api_btn.setObjectName("SettingsMenuButton")
        api_btn.clicked.connect(lambda: self.tts_worker.load_voice(self.custom_api_input.text()))
        
        api_layout.addWidget(self.custom_api_input)
        api_layout.addWidget(api_btn)
        layout.addWidget(api_container)

        return container

    # --------------------------------------------------
    # MODEL CARD
    # --------------------------------------------------
    def _create_model_card(self, m_data: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("ModelCard")

        layout = QVBoxLayout(card)

        header = QHBoxLayout()
        name_lbl = QLabel(m_data["name"])
        header.addWidget(name_lbl)

        header.addStretch()

        size_lbl = QLabel(m_data["size"])
        header.addWidget(size_lbl)

        layout.addLayout(header)

        actions = QHBoxLayout()

        dl_btn = QPushButton("Download")
        dl_btn.clicked.connect(lambda _=None, m_id=m_data["id"]: self._trigger_download(m_id))

        act_btn = QPushButton("Activate")
        act_btn.clicked.connect(
            lambda _=None, path=m_data["path"]: self.tts_worker.load_voice(path)
            if self.tts_worker else None
        )

        actions.addWidget(dl_btn)
        actions.addWidget(act_btn)

        # FIX 1 & 2: Correct operator evaluation and visibility decoupling
        if m_data["id"] in ["f5", "voxtral"]:
            m_id = m_data["id"]

            clone_btn = QPushButton("🎤")
            clone_btn.setToolTip("Upload voice sample")
            
            # Using default args in lambda to prevent late-binding loop issues
            clone_btn.clicked.connect(
                lambda _=None, model_id=m_id: self._pick_clone_voice(model_id)
            )

            # Always show the clone button for applicable models during debugging
            clone_btn.setVisible(True)

            actions.addWidget(clone_btn)
            card.clone_btn = clone_btn

        layout.addLayout(actions)

        card.dl_btn = dl_btn
        card.act_btn = act_btn

        return card
    
    # --------------------------------------------------
    # CLONING FIX
    # --------------------------------------------------
    def _pick_clone_voice(self, m_id):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3-second Voice Sample",
            "",
            "Audio Files (*.wav *.mp3)"
        )

        if not file_path:
            return

        logger.info(f"Cloning voice for {m_id} from {file_path}")

        # FIX 3 & 4: Ensure the correct provider is active BEFORE cloning
        if self.tts_worker:
            target_path = "models/tts/f5/model.safetensors"
            target_provider = "F5Provider"

            # Dynamically handle Voxtral if that was the button clicked
            if m_id == "voxtral":
                target_path = "models/tts/voxtral/model.safetensors"
                target_provider = "VoxtralProvider"

            active_prov = getattr(self.tts_worker, "active_provider", None)
            
            # If no provider is active, or the wrong one is active, load the right one
            if not active_prov or active_prov.__class__.__name__ != target_provider:
                logger.info(f"Activating {target_provider} to handle cloning request...")
                self.tts_worker.load_voice(target_path)

            self.tts_worker.set_voice(file_path)

    def _update_download_ui(self, progress: int, status: str):
        # We need to know which model is downloading. 
        # For this prototype, we'll check the active download model from the worker.
        m_id = self.downloader.model_id
        if m_id in self.model_cards:
            card = self.model_cards[m_id]
            card.progress_bar.show()
            card.progress_bar.setValue(progress)
            card.dl_btn.setEnabled(False)
            card.dl_btn.setText("Downloading...")

    def _on_download_finished(self, m_id: str, success: bool):
        if m_id in self.model_cards:
            card = self.model_cards[m_id]
            card.progress_bar.hide()
            if success:
                card.dl_btn.hide()
                card.act_btn.show()
            else:
                card.dl_btn.setEnabled(True)
                card.dl_btn.setText("Retry Download")

    def _trigger_download(self, m_id: str):
        # 🔑 CORRECTED 'RESOLVE' URLs (No more /blob/ links!)
        urls = {
            "f5": {
                "models/tts/f5/model.safetensors": "https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/model_1250000.safetensors"
            },
            "voxtral": {
                "models/tts/voxtral/model.safetensors": "https://huggingface.co/mistralai/Voxtral-TTS/resolve/main/model.safetensors"
            }
        }
        if m_id in urls:
            self.downloader.start_download(m_id, urls[m_id])

    def _pick_clone_voice(self, m_id):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select 3-second Voice Sample", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            # Tell the worker to use this file as the reference tensor
            self.tts_worker.set_voice(file_path)
            logger.info(f"Cloning voice for {m_id} from {file_path}")

    def _setup_ui(self):
        from PyQt6.QtWidgets import QMenu 
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QLabel("SYSTEM SETTINGS")
        title.setObjectName("ViewTitle")
        main_layout.addWidget(title)

        # Scrollable Area
        scroll = QScrollArea()
        scroll.setObjectName("SettingsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_content.setObjectName("SettingsContent")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(30)

        # --- SECTION 1: AUDIO & HARDWARE ---
        content_layout.addWidget(self._build_section_header("fa5s.microchip", "AUDIO & HARDWARE"))
        
        hw_widget = QWidget()
        hw_widget.setObjectName("SettingsFormContainer")
        
        hw_form = QFormLayout(hw_widget)
        hw_form.setSpacing(15)
        hw_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.mic_selector = QPushButton("Select Input Device...")
        self.device_selector = QPushButton("Select Output Device...")
        
        for btn in [self.mic_selector, self.device_selector]:
            btn.setObjectName("SettingsMenuButton")
            btn.setMaximumWidth(350)
            btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            btn.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
            btn.setMenu(QMenu(btn))

        self.timeout_spinner = QDoubleSpinBox()
        self.timeout_spinner.setFixedWidth(90)
        self.timeout_spinner.setRange(0.5, 5.0)
        self.timeout_spinner.setSingleStep(0.1)
        self.timeout_spinner.setValue(self.audio_worker.silence_timeout if self.audio_worker else 2.0)
        self.timeout_spinner.setSuffix(" sec")
        if self.audio_worker:
            self.timeout_spinner.valueChanged.connect(self.audio_worker.set_silence_timeout)

        self.threshold_spinner = QSpinBox()
        self.threshold_spinner.setFixedWidth(90)
        self.threshold_spinner.setRange(1, 100)
        self.threshold_spinner.setValue(int(self.audio_worker.speech_threshold) if self.audio_worker else 2)
        self.threshold_spinner.setSuffix("%")
        if self.audio_worker:
            self.threshold_spinner.valueChanged.connect(self.audio_worker.set_speech_threshold)

        hw_form.addRow("Audio Input", self.mic_selector)
        hw_form.addRow("Audio Output", self.device_selector)
        hw_form.addRow("Silence Cutoff", self.timeout_spinner)
        hw_form.addRow("Mic Sensitivity", self.threshold_spinner)

        self.pin_audio_cb = QCheckBox("Pin Audio Controls to Toolbar")
        self.pin_audio_cb.toggled.connect(self.audio_pin_toggle.emit)
        hw_form.addRow("", self.pin_audio_cb)

        content_layout.addWidget(hw_widget)
        content_layout.addWidget(self._build_divider())

        # --- SECTION 2: AI MODELS & ROUTING ---
        content_layout.addWidget(self._build_section_header("fa5s.network-wired", "AI MODELS & ROUTING"))
        
        ai_widget = QWidget()
        ai_widget.setObjectName("SettingsFormContainer")
        ai_form = QFormLayout(ai_widget)
        ai_form.setSpacing(15)
        ai_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.wakeword_selector = QPushButton("Select Wakeword...")
        self.provider_selector = QPushButton("Select Provider...")
        self.voice_selector = QPushButton("Select Voice...")

        for btn in [self.wakeword_selector, self.provider_selector, self.voice_selector]:
            btn.setObjectName("SettingsMenuButton")
            btn.setMaximumWidth(250)
            btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            btn.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
            btn.setMenu(QMenu(btn))

        ai_form.addRow("Active Wakeword", self.wakeword_selector)
        ai_form.addRow("AI Provider", self.provider_selector)

        content_layout.addWidget(ai_widget)
        content_layout.addWidget(self._build_divider())
        content_layout.addWidget(self._build_section_header("fa5s.headphones", "TTS ENGINE HUB"))
        
        # This calls the method I gave you in the last turn and adds it to the screen!
        self.tts_hub_widget = self._build_tts_hub()
        content_layout.addWidget(self.tts_hub_widget)
        
        # --- 🔑 SECTION 3: NLP RAG TRIGGERS ---
        content_layout.addWidget(self._build_section_header("fa5s.bolt", "NLP RAG TRIGGERS"))
        content_layout.addWidget(self._build_triggers_manager())

        content_layout.addStretch()
        scroll.setWidget(scroll_content)
        
        # Ensure initial styling is applied
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        self._apply_spinbox_style(is_dark)
        main_layout.addWidget(scroll)

    def _apply_spinbox_style(self, is_dark: bool):
        """Forces borders to be visible on inputs, checkboxes, and the custom trigger elements."""
        border_color = "rgba(255, 255, 255, 0.15)" if is_dark else "#cbd5e1"
        bg_color = "#313244" if is_dark else "#ffffff"
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        check_bg = "#45475a" if is_dark else "#f1f5f9"

        style = f"""
            QDoubleSpinBox, QSpinBox {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 5px 10px;
            }}
            QCheckBox {{ color: {text_color}; font-size: 13px; }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {border_color};
                border-radius: 4px;
                background-color: {check_bg};
            }}
            QCheckBox::indicator:checked {{
                background-color: #8b5cf6; 
                image: url(assets/icons/check_mark.png);
            }}
        """
        self.timeout_spinner.setStyleSheet(style)
        self.threshold_spinner.setStyleSheet(style)
        self.pin_audio_cb.setStyleSheet(style)
        self.auto_activator_cb.setStyleSheet(style)
        
        # 🔑 Style the NLP Trigger input & list
        if hasattr(self, 'trigger_input'):
            self.trigger_input.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {bg_color};
                    color: {text_color};
                    border: 1px solid {border_color};
                    border-radius: 8px;
                    padding: 8px 15px;
                    font-size: 13px;
                }}
            """)
            
        if hasattr(self, 'trigger_list'):
            self.trigger_list.setStyleSheet(f"""
                QListWidget {{
                    background-color: transparent;
                    border: 1px solid {border_color};
                    border-radius: 8px;
                }}
                QListWidget::item {{
                    border-bottom: 1px solid {border_color};
                }}
            """)

    # --------------------------------------------------------- #
    #  🔑 NEW RAG TRIGGER MANAGER                              #
    # --------------------------------------------------------- #

    def _build_triggers_manager(self) -> QWidget:
        """Builds the input box and list UI for custom RAG triggers."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 0, 15, 10)
        layout.setSpacing(15)
        
        # Instruction Label
        instruction = QLabel("Add custom phrases that will trigger a semantic search of your Knowledge Base:")
        instruction.setStyleSheet("color: #64748b; font-size: 12px; font-style: italic;")
        layout.addWidget(instruction)

        # 🔑 NEW: Master Checkbox
        self.auto_activator_cb = QCheckBox("Enable NLP Auto-Activator")
        self.auto_activator_cb.setChecked(True)
        self.auto_activator_cb.toggled.connect(self.auto_activator_toggle.emit)
        layout.addWidget(self.auto_activator_cb)
        
        # Input Row
        input_row = QHBoxLayout()
        self.trigger_input = QLineEdit()
        self.trigger_input.setPlaceholderText("e.g. 'search my notes for...'")
        self.trigger_input.returnPressed.connect(self._on_add_trigger)
        
        self.trigger_add_btn = QPushButton()
        self.trigger_add_btn.setFixedSize(36, 36)
        self.trigger_add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # 🔑 FIX 1: Initialize the icon and CSS immediately upon creation
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
        btn_bg = "#313244" if is_dark else "#e2e8f0"
        btn_hover = "#45475a" if is_dark else "#cbd5e1"
        
        self.trigger_add_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
        self.trigger_add_btn.setStyleSheet(f"""
            QPushButton {{ background: {btn_bg}; border: none; border-radius: 8px; }}
            QPushButton:hover {{ background: {btn_hover}; }}
        """)
        
        self.trigger_add_btn.clicked.connect(self._on_add_trigger)
        
        input_row.addWidget(self.trigger_input)
        input_row.addWidget(self.trigger_add_btn)
        layout.addLayout(input_row)
        
        # Display List
        self.trigger_list = QListWidget()
        # 🔑 FIX 2: Force the layout engine to respect a minimum height!
        self.trigger_list.setMinimumHeight(320) 
        
        self.trigger_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.trigger_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        layout.addWidget(self.trigger_list)
        
        self._refresh_trigger_list()
        
        return container

    def _refresh_trigger_list(self):
        """Pulls from SQLite and rebuilds the styled list."""
        if not hasattr(self, 'trigger_list'): return
        
        self.trigger_list.clear()
        triggers = self.db.get_rag_triggers()
        
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        icon_color = "#ef4444" # Danger Red for Trash
        hover_bg = "rgba(239, 68, 68, 0.1)" # Faint red hover
        
        for phrase in triggers:
            item = QListWidgetItem()
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(15, 5, 10, 5)
            
            lbl = QLabel(phrase)
            lbl.setStyleSheet(f"color: {text_color}; font-size: 13px; font-weight: bold;")
            
            del_btn = QPushButton()
            del_btn.setIcon(qta.icon('fa5s.trash-alt', color=icon_color))
            del_btn.setFixedSize(28, 28)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 4px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)
            del_btn.clicked.connect(lambda checked, p=phrase: self._on_delete_trigger(p))
            
            layout.addWidget(lbl)
            layout.addStretch()
            layout.addWidget(del_btn)
            
            item.setSizeHint(QSize(0, 60))
            self.trigger_list.addItem(item)
            self.trigger_list.setItemWidget(item, row)

    def _on_add_trigger(self):
        text = self.trigger_input.text().strip()
        if text:
            # Add to database
            success = self.db.add_rag_trigger(text)
            if success:
                self.trigger_input.clear()
                self._refresh_trigger_list()
                
    def _on_delete_trigger(self, phrase):
        # Remove from database
        self.db.remove_rag_trigger(phrase)
        self._refresh_trigger_list()


    # --------------------------------------------------------- #
    #  THE PRESTIGE MENU LOGIC                                  #
    # --------------------------------------------------------- #
    def _build_prestige_menu(self, button, items, callback):
        from PyQt6.QtWidgets import QMenu, QWidgetAction, QListWidget
        from PyQt6.QtCore import Qt

        menu = QMenu(button)
        menu.setObjectName("PrestigeMenu")
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        self._apply_menu_theme(menu, is_dark)

        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        for label, data in items:
            list_widget.addItem(label)
            
        required_height = len(items) * 32 + 10 
        main_win = self.window()
        max_height = int(main_win.height() * 0.5) if main_win else 400
        list_widget.setFixedHeight(min(required_height, max_height))

        def sync_dropdown_width():
            list_widget.setFixedWidth(button.width() - 8)

        menu.aboutToShow.connect(sync_dropdown_width)

        def on_item_clicked(item):
            selected_label = item.text()
            matched_data = next((d for l, d in items if l == selected_label), selected_label)
            self._handle_selection(button, selected_label, matched_data, callback)
            menu.hide()

        list_widget.itemClicked.connect(on_item_clicked)

        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)
        button.setMenu(menu)

    def _apply_menu_theme(self, menu, is_dark: bool):
        from PyQt6.QtGui import QPalette, QColor

        palette = QPalette()
        if is_dark:
            bg      = QColor("#1e1e2e")
            fg      = QColor("#cdd6f4")
            sel_bg  = QColor("#313244")
            sel_fg  = QColor("#cdd6f4")
            border  = "rgba(255, 255, 255, 0.1)"
            hover   = "#313244"
        else:
            bg      = QColor("#ffffff")
            fg      = QColor("#1e293b")
            sel_bg  = QColor("#f1f5f9")
            sel_fg  = QColor("#0f172a")
            border  = "#cbd5e1"
            hover   = "#f1f5f9"

        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Highlight, sel_bg)
        palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)

        menu.setPalette(palette)
        menu.setStyleSheet(f"""
            QMenu {{ background-color: {bg.name()}; border: 1px solid {border}; border-radius: 6px; padding: 4px; }}
            QListWidget#PrestigeMenuList {{ background-color: transparent; border: none; outline: none; }}
            QListWidget#PrestigeMenuList::item {{ background-color: transparent; color: {fg.name()}; padding: 8px 25px; border-radius: 4px; min-height: 24px; }}
            QListWidget#PrestigeMenuList::item:selected, QListWidget#PrestigeMenuList::item:hover {{ background-color: {hover}; color: {sel_fg.name()}; }}
            QScrollBar:vertical {{ border: none; background: transparent; width: 6px; margin: 0px; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 3px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

    def refresh_menu_themes(self, is_dark: bool):
        """Standardizes icons and borders when the theme is toggled."""
        buttons = [self.mic_selector, self.device_selector, self.wakeword_selector, self.provider_selector, self.voice_selector]
        for btn in buttons:
            if btn.menu():
                self._apply_menu_theme(btn.menu(), is_dark)

        # Update Section Header Icons
        icon_color = "#8b5cf6" if is_dark else "#4c4f69" 
        
        for icon_lbl in [getattr(self, 'audio_icon_label', None), getattr(self, 'ai_icon_label', None), getattr(self, 'rag_icon_label', None)]:
            if icon_lbl:
                name = icon_lbl.property("icon_name")
                icon_lbl.setPixmap(qta.icon(name, color=icon_color).pixmap(QSize(18, 18)))

        # Update Trigger Add Button
        if hasattr(self, 'trigger_add_btn'):
            btn_bg = "#313244" if is_dark else "#e2e8f0"
            btn_hover = "#45475a" if is_dark else "#cbd5e1"
            self.trigger_add_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
            self.trigger_add_btn.setStyleSheet(f"""
                QPushButton {{ background: {btn_bg}; border: none; border-radius: 8px; }}
                QPushButton:hover {{ background: {btn_hover}; }}
            """)

        self._apply_spinbox_style(is_dark)
        self._refresh_trigger_list() # Repaints the list fonts & trash icons!

    def _handle_selection(self, button, label, data, callback):
        button.setText(label)
        callback(data)

    def _build_section_header(self, icon_name, title_text):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        icon_label.setProperty("icon_name", icon_name)
        
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
        icon_label.setPixmap(qta.icon(icon_name, color=icon_color).pixmap(QSize(18, 18)))
        icon_label.setProperty("class", "SectionHeaderIcon")
        
        if "AUDIO" in title_text: self.audio_icon_label = icon_label
        elif "MODELS" in title_text: self.ai_icon_label = icon_label
        elif "TRIGGERS" in title_text: self.rag_icon_label = icon_label
        
        text_label = QLabel(title_text)
        text_label.setProperty("class", "SectionHeaderLabel")
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        return container

    def _build_divider(self):
        line = QFrame()
        line.setObjectName("SettingsDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _populate_hardware_selectors(self):
        mics = get_input_devices() 
        if mics:
            self._build_prestige_menu(self.mic_selector, [(name, idx) for idx, name in mics], lambda idx: self.audio_worker.set_input_device(idx) if self.audio_worker else None)
            active_mic_name = mics[0][1] 
            if self.audio_worker and hasattr(self.audio_worker, 'input_device_index'):
                for idx, name in mics:
                    if idx == self.audio_worker.input_device_index:
                        active_mic_name = name; break
            self.mic_selector.setText(active_mic_name)

        outputs = get_output_devices()
        if outputs:
            self._build_prestige_menu(self.device_selector, [(name, idx) for idx, name in outputs], lambda idx: self.tts_worker.set_device(idx) if self.tts_worker else None)
            active_output_name = outputs[0][1]
            if self.tts_worker and hasattr(self.tts_worker, 'current_device_index'):
                for idx, name in outputs:
                    if idx == self.tts_worker.current_device_index:
                        active_output_name = name; break
            self.device_selector.setText(active_output_name)

        if self.audio_worker:
            wakewords = list(self.audio_worker.available_wakewords.keys())
            if wakewords:
                self._build_prestige_menu(self.wakeword_selector, [(w, w) for w in wakewords], self.audio_worker.set_wakeword)
                if hasattr(self.audio_worker, 'active_wakeword_name') and self.audio_worker.active_wakeword_name:
                    self.wakeword_selector.setText(self.audio_worker.active_wakeword_name)
                else:
                    self.wakeword_selector.setText(wakewords[0])

        providers = [("Ollama (Port 11434)", 11434), ("LM Studio (Port 1234)", 1234)]
        self._build_prestige_menu(self.provider_selector, providers, lambda port: self.llm_worker.set_provider(port) if self.llm_worker else None)
        
        if is_port_open(1234): self.provider_selector.setText("LM Studio (Port 1234)")
        elif is_port_open(11434): self.provider_selector.setText("Ollama (Port 11434)")

    def update_voice_dropdown(self, model_name: str, voices: list) -> None:
        if not voices: return
        self._build_prestige_menu(self.voice_selector, [(v, v) for v in voices], lambda v: self.tts_worker.set_voice(v) if self.tts_worker else None)
        self.voice_selector.setText(voices[0])
        if self.tts_worker: self.tts_worker.set_voice(voices[0])