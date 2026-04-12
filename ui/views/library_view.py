from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QPushButton, QListWidget, QTextEdit, QFileDialog, QMessageBox, QSizePolicy,
    QProgressBar, QLineEdit
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
import qtawesome as qta
from pathlib import Path
from ui.components.prestige_dialog import PrestigeDialog
import logging

logger = logging.getLogger("Qube.UI.Library")

class LibraryView(QWidget):
    ingest_requested = pyqtSignal(list)

    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        # We need the vector store for reconstruction and deep deletion
        # (Assuming 'store' was added to the workers dictionary in main.py)
        self.store = workers.get("store") 

        # 🔑 THE FIX: Declare the flag here so it exists on boot!
        self._had_ingestion_error = False
        
        self.active_filename = None
        self._setup_ui()
        self.refresh_library_list()
        # Rows are built before this view is parented under MainWindow; QSS + selection need a pass once attached.
        QTimer.singleShot(0, self._update_row_colors)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # --- COLUMN 1: Document List Sidebar ---
        self.list_pane = self._build_list_pane()
        layout.addWidget(self.list_pane)

        # --- COLUMN 2: Document Preview Stage ---
        self.preview_stage = self._build_preview_stage()

        # 🔑 THE FIX: Force the layout to ignore the internal text's size demands
        self.preview_stage.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        layout.addWidget(self.preview_stage, stretch=1)

        # Forces the button to load with the default Dark Mode purple on startup
        self.refresh_button_themes(is_dark=True)

    def _build_list_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(280)
        frame.setObjectName("LibrarySidebar") 
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        # --- HEADER AREA ---
        header_layout = QHBoxLayout()
        self.list_title = QLabel("KNOWLEDGE BASE")
        self.list_title.setProperty("class", "SidebarTitle")
        
        self.add_btn = QPushButton()
        self.add_btn.setIcon(qta.icon('fa5s.plus')) 
        self.add_btn.setProperty("class", "IconButton") 
        self.add_btn.setToolTip("Ingest New Document")
        self.add_btn.clicked.connect(self._browse_for_document) 
        
        header_layout.addWidget(self.list_title)
        header_layout.addStretch()
        header_layout.addWidget(self.add_btn)
        
        layout.addLayout(header_layout)

        # --- PROGRESS BAR (Moved here, right under the header!) ---
        self.ingest_progress = QProgressBar()
        self.ingest_progress.setObjectName("IngestProgressBar")
        self.ingest_progress.setRange(0, 100)
        self.ingest_progress.setFixedHeight(4) # Made it slightly thinner for a sleeker look
        self.ingest_progress.setTextVisible(False)
        self.ingest_progress.hide() 
        layout.addWidget(self.ingest_progress)

        # The Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search documents...")
        self.search_bar.setObjectName("LibrarySearchBar")
        self.search_bar.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_bar)

        # Document List
        self.doc_list = QListWidget()
        self.doc_list.setObjectName("LibraryDocList")
        self.doc_list.itemClicked.connect(self._on_document_selected)
        self.doc_list.itemSelectionChanged.connect(self._update_row_colors)
        
        # 🔑 NEW: Wire up the scrollbar for infinite scrolling
        self.library_offset = 0
        self.is_loading_library = False
        self.doc_list.verticalScrollBar().valueChanged.connect(self._on_library_scroll)

        layout.addWidget(self.doc_list)

        return frame
    
    def _build_preview_stage(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LibraryPreviewStage")

        # 🔑 FIX 1: Strip any global card styling from the main frame
        frame.setStyleSheet("background: transparent; border: none;")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Header Area for Preview
        header_layout = QHBoxLayout()
        self.doc_title = QLabel("No Document Selected")
        self.doc_title.setObjectName("PreviewDocTitle")
        
        # 🔑 FIX 1: Stop the title from forcing the window wider on long filenames
        self.doc_title.setWordWrap(True)
        self.doc_title.setMinimumWidth(0)
        self.doc_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        self.doc_stats = QLabel("")
        self.doc_stats.setObjectName("PreviewStatsText")
        
        # 🔑 FIX 2: Stop the stats from stretching the window
        self.doc_stats.setWordWrap(True)
        self.doc_stats.setMinimumWidth(0)
        self.doc_stats.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        title_vbox = QVBoxLayout()
        title_vbox.addWidget(self.doc_title)
        title_vbox.addWidget(self.doc_stats)
        
        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Reconstructed Text Area
        self.text_preview = QTextEdit()
        self.text_preview.setObjectName("DocumentPreviewArea")
        self.text_preview.setReadOnly(True)
        
        # --- THE FIX: Aggressive Wrapping & Shrink Allowance ---
        from PyQt6.QtGui import QTextOption
        
        # 1. Allow the widget to shrink freely when the user resizes the app
        self.text_preview.setMinimumWidth(0) 
        self.text_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 2. Force wrapping strictly at the widget's edge
        self.text_preview.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        
        # 3. CRITICAL: Break long unbreakable strings (like PDF hashes or long titles) 
        # instead of stretching the parent window.
        self.text_preview.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        
        # 4. Strip the default PyQt sunken card box for that clean look we discussed
        self.text_preview.setStyleSheet("background: transparent; border: none;")

        self.text_preview.setPlaceholderText("Select a document from the left to view its contents.")
        layout.addWidget(self.text_preview)

        return frame

    # --------------------------------------------------------- #
    #  LOGIC WIRING                                             #
    # --------------------------------------------------------- #

    def _apply_menu_theme(self, menu, is_dark: bool):
        """Standardizes the menu appearance with Prestige rounding and colors."""
        
        # THIS IS THE MAGIC LINE TO KILL THE GHOST SQUARE
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        bg, fg, hover = ("#1e1e2e", "#cdd6f4", "#313244") if is_dark else ("#ffffff", "#1e293b", "#f1f5f9")
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"

        menu.setStyleSheet(f"""
            QMenu {{ background-color: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 12px; padding: 5px; }}
            QMenu::item {{ background-color: transparent; padding: 8px 25px; border-radius: 8px; }}
            QMenu::item:selected {{ background-color: {hover}; color: {fg}; }}
        """)

    def refresh_library_list(self):
        """Clears the list, resets the offset, and pulls the first batch."""
        self.doc_list.clear()
        self.library_offset = 0
        self.is_loading_library = False

        count = self.db.get_document_count()
        display_count = "999+" if count > 999 else str(count)
        
        if hasattr(self, 'list_title'):
            self.list_title.setText(f"KNOWLEDGE BASE ({display_count})")
            
        # Load the initial batch!
        self._load_library_batch()

    def _on_library_scroll(self, value):
        """Triggered every time the user scrolls."""
        bar = self.doc_list.verticalScrollBar()
        # If the scrollbar hits the absolute maximum, load the next batch
        if value == bar.maximum():
            self._load_library_batch()

    def _load_library_batch(self):
        """Fetches the next chunk of documents and appends it to the list."""
        if getattr(self, 'is_loading_library', False):
            return 
            
        self.is_loading_library = True

        from PyQt6.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
        from PyQt6.QtCore import Qt, QSize
        import qtawesome as qta
        
        # Match ConversationsView._load_history_batch: window may be None until after MainWindow adds this widget.
        is_dark = True
        main_win = self.window()
        if main_win and hasattr(main_win, "_is_dark_theme"):
            is_dark = main_win._is_dark_theme

        t_color = "#cdd6f4" if is_dark else "#1e293b"
        icon_color = "#6c7086" if is_dark else "#64748b" 
        
        # 🔑 THE FIX: Pass both limit and offset to the DB
        try:
            docs = self.db.get_library_documents(limit=20, offset=self.library_offset)
        except TypeError:
            # Fallback if DB isn't updated yet
            docs = self.db.get_library_documents()
            if self.library_offset > 0:
                docs = [] 

        if not docs:
            self.is_loading_library = False
            return

        for doc in docs:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, doc)
            
            row = QWidget()
            row.setObjectName("HistoryRowWidget")
            row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            
            lay = QHBoxLayout(row)
            lay.setContentsMargins(15, 0, 10, 0)
            lay.setSpacing(10)
            
            # 1. The Title
            item_text = f"{doc['filename']} ({doc['file_size_kb']} KB)"
            lbl = QLabel(item_text)
            lbl.setObjectName("HistoryRowTitle")
            lbl.setStyleSheet(f"color: {t_color}; background: transparent; border: none; font-size: 13px; font-weight: 500;")
            
            # 2. The Kebab Button
            btn = QPushButton()
            btn.setObjectName("HistoryOptionsBtn")
            btn.setFixedSize(28, 28)
            btn.setIcon(qta.icon('fa5s.ellipsis-v', color=icon_color))
            btn.setIconSize(QSize(16, 16)) 
            btn.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0px; } QPushButton { border: none; background: transparent; }")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            
            # 3. The Menu
            menu = QMenu(btn)
            if hasattr(self, '_apply_menu_theme'):
                self._apply_menu_theme(menu, is_dark)
            
            rename_action = menu.addAction(qta.icon('fa5s.edit', color='#89b4fa'), "Rename Document")
            rename_action.triggered.connect(lambda _, fname=doc['filename']: self._trigger_rename_document(fname))
            
            menu.addSeparator()

            delete_action = menu.addAction(qta.icon('fa5s.trash-alt', color='#ef4444'), "Delete Document")
            delete_action.triggered.connect(lambda _, fname=doc['filename']: self._trigger_delete_document(fname))
            
            btn.setMenu(menu)
            
            lay.addWidget(lbl)
            lay.addStretch()
            lay.addWidget(btn)
            
            item.setSizeHint(QSize(0, 45))
            self.doc_list.addItem(item)
            self.doc_list.setItemWidget(item, row)

        # 🔑 Increment the offset for the next scroll
        self.library_offset += 20
        self.is_loading_library = False

    def _update_row_colors(self):
        """Forces text color changes since Qt CSS cannot pass :selected states to setItemWidget."""
        from PyQt6.QtWidgets import QLabel
        
        # 1. Safely Detect Theme
        is_dark = getattr(self.window(), '_is_dark_theme', True)
            
        # 2. 🔑 THE FIX: The Colors!
        # Unselected: Light gray in Dark Mode, Slate in Light Mode
        normal_color = "#cdd6f4" if is_dark else "#1e293b"
        # Selected: The background bubble is solid, so text should ALWAYS be White
        selected_color = "#ffffff" 

        # 3. Target whichever list is in this specific file
        target_list = getattr(self, 'doc_list', getattr(self, 'history_list', None))
        if not target_list: 
            return

        # 4. Loop through and forcefully apply the correct color
        for i in range(target_list.count()):
            item = target_list.item(i)
            widget = target_list.itemWidget(item)
            if widget:
                lbl = widget.findChild(QLabel, "HistoryRowTitle")
                if lbl:
                    color = selected_color if item.isSelected() else normal_color
                    lbl.setStyleSheet(f"color: {color}; background: transparent; border: none; font-size: 13px; font-weight: 500;")

    def _trigger_rename_document(self, old_filename):
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        dlg = PrestigeDialog(self, "Rename Document", f"Enter a new name for '{old_filename}':", is_dark, is_input=True, default_text=old_filename)
        
        if dlg.exec() and dlg.result_text and dlg.result_text.strip():
            new_name = dlg.result_text.strip()
            
            # 1. Update SQLite
            self.db.rename_document_metadata(old_filename, new_name)
            
            # 2. Update Vector Store (CRITICAL: You must implement this in your LanceDB class!)
            if self.store and hasattr(self.store, 'rename_document'):
                self.store.rename_document(old_filename, new_name)
            elif self.store:
                logger.warning(f"Renamed {old_filename} in SQLite, but 'rename_document' is missing in Vector Store!")

            # 3. Update UI if they renamed the currently open document
            if self.active_filename == old_filename:
                self.active_filename = new_name
                self.doc_title.setText(new_name)
                
            self.refresh_library_list()

    def _trigger_delete_document(self, filename):
        """Spawns the Prestige dialog and coordinates deletion from both DBs."""
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        dlg = PrestigeDialog(self, "Delete Document", f"Are you sure you want to permanently delete and un-index '{filename}'?", is_dark)
        
        if dlg.exec():
            logger.info(f"User initiated deletion of {filename}")
            
            if self.store:
                self.store.delete_document(filename)
            
            self.db.delete_document_metadata(filename)
            
            # If they deleted the document they are currently looking at, clear the preview
            if self.active_filename == filename:
                self.active_filename = None
                self.doc_title.setText("No Document Selected")
                self.doc_stats.setText("")
                self.text_preview.setHtml("<center><h3>Document deleted.</h3></center>")
                
            self.refresh_library_list()

    def _filter_list(self, text: str):
        """Hides/Shows list items based on the search bar text."""
        search_term = text.lower()
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            # Retrieve the filename from the stored metadata
            doc_data = item.data(Qt.ItemDataRole.UserRole)
            if doc_data and search_term in doc_data['filename'].lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    def _on_document_selected(self, item):
        doc_data = item.data(Qt.ItemDataRole.UserRole)
        self.active_filename = doc_data['filename']
        
        self.doc_title.setText(self.active_filename)
        self.doc_stats.setText(f"Size: {doc_data['file_size_kb']} KB | Chunks Indexed: {doc_data['chunk_count']}")

        self.text_preview.setHtml("<center><h3>Reconstructing document from vector space...</h3></center>")
        
        # Pull chunks from LanceDB and stitch them together
        if self.store:
            content = self.store.reconstruct_document(self.active_filename)
            self.text_preview.setPlainText(content)
        else:
            self.text_preview.setPlainText("Error: Vector store not connected.")

    def _browse_for_document(self):
        """Opens a file dialog, checks for duplicates, and handles overwrites."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents to Ingest", "", "Documents (*.txt *.md *.pdf *.epub)"
        )
        if not files:
            return

        paths = [Path(f) for f in files]
        
        # 1. Check if any selected files already exist in our SQLite registry
        existing_files = []
        current_docs = [doc['filename'] for doc in self.db.get_library_documents()]
        
        for p in paths:
            if p.name in current_docs:
                existing_files.append(p.name)

        # 2. Prompt the user if duplicates are found
        if existing_files:
            msg = (f"The following {len(existing_files)} file(s) already exist in your Knowledge Base:\n\n"
                   f"{', '.join(existing_files[:5])}" + ("..." if len(existing_files) > 5 else "") +
                   "\n\nDo you want to overwrite them?")
            
            # 🔑 Use PrestigeDialog for the Yes/No check
            is_dark = getattr(self.window(), '_is_dark_theme', True)
            dialog = PrestigeDialog(self, "Overwrite Files?", msg, is_dark)

            # .exec() returns truthy if they clicked the primary confirmation button
            if dialog.exec():
                logger.info("User chose to overwrite existing files. Purging old data...")
                for name in existing_files:
                    if self.store: self.store.delete_document(name)
                    self.db.delete_document_metadata(name)
            else:
                paths = [p for p in paths if p.name not in existing_files]
                if not paths:
                    logger.info("Ingestion cancelled; all selected files were duplicates and user declined overwrite.")
                    return

        # 3. Proceed with the standard ingestion UI updates
        self.ingest_progress.setValue(0)
        self.ingest_progress.show()
        self.add_btn.setEnabled(False)
        # REMOVED: self.add_btn.setText(...)
        
        logger.info(f"Emitting {len(paths)} files to main pipeline for ingestion.")
        # 🔑 Reset the flag right before starting a new job
        self._had_ingestion_error = False
        self.ingest_requested.emit(paths)

    # --- UI Receivers for Worker Progress ---
    def update_ingestion_progress(self, percent: int):
        self.ingest_progress.setValue(percent)

    def show_error(self, error_msg: str):
        """Displays ingestion errors to the user and resets the UI."""
        self._had_ingestion_error = True 
        
        self.ingest_progress.hide()
        self.ingest_progress.setValue(0)
        self.add_btn.setEnabled(True)

        # 🔑 Use the superior PrestigeDialog
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        dialog = PrestigeDialog(self, "Ingestion Failed", str(error_msg), is_dark)
        dialog.exec()

    def complete_ingestion(self, total_chunks: int):
        self.ingest_progress.hide()
        self.ingest_progress.setValue(0)
        self.add_btn.setEnabled(True)
        
        if self._had_ingestion_error:
            return 
            
        self.refresh_library_list()
        
        if total_chunks == 0:
            is_dark = getattr(self.window(), '_is_dark_theme', True)
            msg = "Process finished, but 0 chunks were added. This usually means the file was already in the database, or it is a scanned PDF with no readable text."
            dialog = PrestigeDialog(self, "No Data Added", msg, is_dark)
            dialog.exec()

    def refresh_button_themes(self, is_dark: bool):
        """Dynamically updates the color of the Add Document button."""
        import qtawesome as qta
        
        # Icon color: Catppuccin Purple in Dark Mode, Deep Slate in Light Mode
        icon_color = "#8b5cf6" if is_dark else "#1e293b"
        
        # Subtle hover background
        hover_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.05)"
        
        if hasattr(self, 'add_btn'):
            self.add_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
            self.add_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)