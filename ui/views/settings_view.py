import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QPushButton,
    QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QProgressBar,
    QStyledItemDelegate, QListView, QMenu
)
from PyQt6.QtCore import Qt, QSize
from pathlib import Path
import logging

from core.audio_utils import get_input_devices, get_output_devices
from core.network import is_port_open

logger = logging.getLogger("Qube.UI.Settings")

class SettingsView(QWidget):
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.audio_worker = workers.get("audio")
        self.tts_worker = workers.get("tts")
        self.llm_worker = workers.get("llm")

        self._setup_ui()
        self._populate_hardware_selectors()

    def _setup_ui(self):
        from PyQt6.QtWidgets import QMenu # Ensure QMenu is imported
        
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
        content_layout.setContentsMargins(0, 0, 0, 0) # Keep zeroed for alignment
        content_layout.setSpacing(30)

        # --- SECTION 1: AUDIO & HARDWARE ---
        content_layout.addWidget(self._build_section_header("fa5s.microchip", "AUDIO & HARDWARE"))
        
        hw_widget = QWidget()
        hw_widget.setObjectName("SettingsFormContainer")
        hw_form = QFormLayout(hw_widget)
        hw_form.setSpacing(15)
        hw_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # We use Buttons instead of ComboBoxes
        self.mic_selector = QPushButton("Select Input Device...")
        self.device_selector = QPushButton("Select Output Device...")
        
        # Styling the buttons to look like prestige dropdowns
        for btn in [self.mic_selector, self.device_selector]:
            btn.setObjectName("SettingsMenuButton")
            btn.setMaximumWidth(350)
            # This puts the chevron on the right side
            btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            btn.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
            # Initialize with an empty menu
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
        content_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    # --- THE PRESTIGE MENU LOGIC ---
    def _build_prestige_menu(self, button, items, callback):
        """Builds a palette-forced QMenu with a dynamic, scrollable list."""
        from PyQt6.QtWidgets import QMenu, QWidgetAction, QListWidget
        from PyQt6.QtCore import Qt

        menu = QMenu(button)
        menu.setObjectName("PrestigeMenu")
        
        # Apply the theme palette
        is_dark = self._is_dark_theme if hasattr(self, '_is_dark_theme') else getattr(self.window(), '_is_dark_theme', True)
        self._apply_menu_theme(menu, is_dark)

        # 1. Create the Scrollable List
        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        
        # --- BUG 2 FIX: Kill the phantom horizontal scrollbar ---
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 2. Populate the List
        for label, data in items:
            list_widget.addItem(label)
            
        # 3. Dynamic Height Calculation
        required_height = len(items) * 32 + 10 
        main_win = self.window()
        max_height = int(main_win.height() * 0.5) if main_win else 400
        list_widget.setFixedHeight(min(required_height, max_height))

        # --- BUG 1 FIX: Just-In-Time Sizing ---
        # This recalculates the exact width a millisecond before the popup opens.
        def sync_dropdown_width():
            # button.width() gets the actual drawn size.
            # We subtract 8px to account for the 4px CSS padding on each side of the QMenu.
            list_widget.setFixedWidth(button.width() - 8)

        menu.aboutToShow.connect(sync_dropdown_width)

        # 4. Handle Selection
        def on_item_clicked(item):
            selected_label = item.text()
            matched_data = next((d for l, d in items if l == selected_label), selected_label)
            self._handle_selection(button, selected_label, matched_data, callback)
            menu.hide()

        list_widget.itemClicked.connect(on_item_clicked)

        # 5. Embed the List into the Menu
        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)

        button.setMenu(menu)

    def _apply_menu_theme(self, menu, is_dark: bool):
        """
        Applies BOTH a QPalette and a stylesheet to a QMenu.
        The palette overrides the OS-level colors; the stylesheet handles
        hover/selection states that the palette alone can't reach.
        """
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

        # Force palette at the window level — this beats the OS
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Highlight, sel_bg)
        palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)

        menu.setPalette(palette)

        # Stylesheet handles padding, radius, and hover states
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {bg.name()};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px;
            }}
            /* Style the embedded list */
            QListWidget#PrestigeMenuList {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget#PrestigeMenuList::item {{
                background-color: transparent;
                color: {fg.name()};
                padding: 8px 25px;
                border-radius: 4px;
                min-height: 24px;
            }}
            QListWidget#PrestigeMenuList::item:selected, 
            QListWidget#PrestigeMenuList::item:hover {{
                background-color: {hover};
                color: {sel_fg.name()};
            }}
            
            /* Sleek internal scrollbar */
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {border};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def refresh_menu_themes(self, is_dark: bool):
        """
        Call this from your main window's _toggle_theme method after
        the app stylesheet has been swapped.  Re-applies palettes to every
        live menu that is already attached to a button.
        """
        buttons = [
            self.mic_selector,
            self.device_selector,
            self.wakeword_selector,
            self.provider_selector,
            self.voice_selector,
        ]
        for btn in buttons:
            m = btn.menu()
            if m:
                self._apply_menu_theme(m, is_dark)


    def _handle_selection(self, button, label, data, callback):
        """Updates UI text and fires the worker logic."""
        button.setText(label)
        callback(data)

    def _on_menu_action(self, button, text, callback):
        button.setText(text)
        callback(text)

    def _build_section_header(self, icon_name, title_text):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        # We remove the hardcoded color here so it can be styled via CSS or default to text color
        icon_label.setPixmap(qta.icon(icon_name).pixmap(QSize(18, 18)))
        icon_label.setProperty("class", "SectionHeaderIcon")
        
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

    # --- HARDWARE POPULATION LOGIC (Menu-Based) ---
    def _populate_hardware_selectors(self):
        # 1. Microphones
        mics = get_input_devices() # Returns [(idx, name)]
        if mics:
            self._build_prestige_menu(self.mic_selector, [(name, idx) for idx, name in mics], self._on_mic_changed)
            
            active_mic_name = mics[0][1] # Fallback to the first item
            if self.audio_worker and hasattr(self.audio_worker, 'input_device_index'):
                active_idx = self.audio_worker.input_device_index
                # Search the list for the name that matches the active index
                for idx, name in mics:
                    if idx == active_idx:
                        active_mic_name = name
                        break
            
            self.mic_selector.setText(active_mic_name)

        # 2. Output Devices
        outputs = get_output_devices() # Returns [(idx, name)]
        if outputs:
            self._build_prestige_menu(self.device_selector, [(name, idx) for idx, name in outputs], self._on_audio_device_changed)
            
            active_output_name = outputs[0][1] # Fallback
            if self.tts_worker and hasattr(self.tts_worker, 'current_device_index'):
                active_idx = self.tts_worker.current_device_index
                for idx, name in outputs:
                    if idx == active_idx:
                        active_output_name = name
                        break
                        
            self.device_selector.setText(active_output_name)

        # 3. Wakewords 
        if self.audio_worker:
            wakewords = list(self.audio_worker.available_wakewords.keys())
            if wakewords:
                self._build_prestige_menu(self.wakeword_selector, [(w, w) for w in wakewords], self.audio_worker.set_wakeword)
                # Correctly pulling the active wakeword from your audio_worker.py
                if hasattr(self.audio_worker, 'active_wakeword_name') and self.audio_worker.active_wakeword_name:
                    self.wakeword_selector.setText(self.audio_worker.active_wakeword_name)
                else:
                    self.wakeword_selector.setText(wakewords[0])

        # 4. Providers 
        providers = [
            ("Ollama (Port 11434)", 11434),
            ("LM Studio (Port 1234)", 1234)
        ]
        self._build_prestige_menu(self.provider_selector, providers, self._on_provider_changed)
        
        if is_port_open(1234):
            self.provider_selector.setText("LM Studio (Port 1234)")
        elif is_port_open(11434):
            self.provider_selector.setText("Ollama (Port 11434)")

    def update_voice_dropdown(self, model_name: str, voices: list) -> None:
        """Receives voices from TTS worker and builds the prestige menu."""
        if not voices:
            return
            
        self._build_prestige_menu(
            self.voice_selector, 
            [(v, v) for v in voices], 
            lambda v: self.tts_worker.set_voice(v)
        )
        # Set the default display text to the first voice
        self.voice_selector.setText(voices[0])
        if self.tts_worker:
            self.tts_worker.set_voice(voices[0])

    # --- UPDATED SIGNAL RECEIVERS ---
    def _on_mic_changed(self, device_index: int):
        if self.audio_worker:
            self.audio_worker.set_input_device(device_index)

    def _on_audio_device_changed(self, device_index: int):
        if self.tts_worker:
            self.tts_worker.set_device(device_index)

    def _on_provider_changed(self, port: int):
        if self.llm_worker:
            self.llm_worker.set_provider(port)