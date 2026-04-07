from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QPushButton, QListWidget, QTextEdit, QFileDialog, QMessageBox, QSizePolicy,
    QProgressBar, QLineEdit
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
import qtawesome as qta
from pathlib import Path
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
        
        self.active_filename = None
        self._setup_ui()
        self.refresh_library_list()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # --- COLUMN 1: Document List Sidebar ---
        self.list_pane = self._build_list_pane()
        layout.addWidget(self.list_pane)

        # --- COLUMN 2: Document Preview Stage ---
        self.preview_stage = self._build_preview_stage()
        layout.addWidget(self.preview_stage, stretch=1)

    def _build_list_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(280)
        frame.setObjectName("LibrarySidebar") 
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        # --- HEADER AREA (Unified with Conversations View) ---
        header_layout = QHBoxLayout()
        title = QLabel("KNOWLEDGE BASE")
        title.setProperty("class", "SidebarTitle")
        
        self.add_btn = QPushButton()
        self.add_btn.setIcon(qta.icon('fa5s.plus')) # Use the identical icon
        self.add_btn.setProperty("class", "IconButton") # Apply the transparent rounded rect style
        self.add_btn.setToolTip("Ingest New Document")
        self.add_btn.clicked.connect(self._browse_for_document) 
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.add_btn)
        
        layout.addLayout(header_layout)

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
        layout.addWidget(self.doc_list)

        # Progress Bar
        self.ingest_progress = QProgressBar()
        self.ingest_progress.setObjectName("IngestProgressBar")
        self.ingest_progress.setRange(0, 100)
        self.ingest_progress.setFixedHeight(6)
        self.ingest_progress.setTextVisible(False)
        self.ingest_progress.hide() 
        layout.addWidget(self.ingest_progress)

        return frame
    
    def _build_preview_stage(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LibraryPreviewStage")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Header Area for Preview
        header_layout = QHBoxLayout()
        self.doc_title = QLabel("No Document Selected")
        self.doc_title.setObjectName("PreviewDocTitle")
        
        self.doc_stats = QLabel("")
        self.doc_stats.setProperty("class", "PreviewStatsText")
        
        self.delete_btn = QPushButton(" Delete File")
        self.delete_btn.setIcon(qta.icon('fa5s.trash-alt'))
        self.delete_btn.setProperty("class", "DangerButton")
        self.delete_btn.hide() 
        self.delete_btn.clicked.connect(self._delete_active_document)

        title_vbox = QVBoxLayout()
        title_vbox.addWidget(self.doc_title)
        title_vbox.addWidget(self.doc_stats)
        
        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        header_layout.addWidget(self.delete_btn)
        layout.addLayout(header_layout)

        # Reconstructed Text Area
        self.text_preview = QTextEdit()
        self.text_preview.setObjectName("DocumentPreviewArea")
        self.text_preview.setReadOnly(True)
        self.text_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # We use a placeholder that will inherit the theme's text color
        self.text_preview.setPlaceholderText("Select a document from the left to view its contents.")
        layout.addWidget(self.text_preview)

        return frame

    # --------------------------------------------------------- #
    #  LOGIC WIRING                                             #
    # --------------------------------------------------------- #

    def refresh_library_list(self):
        """Pulls the registry from SQLite (fast) without touching vectors."""
        self.doc_list.clear()
        docs = self.db.get_library_documents()
        for doc in docs:
            item_text = f"{doc['filename']} ({doc['file_size_kb']} KB)"
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, doc) # Store full metadata dict
            self.doc_list.addItem(item)

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
        self.delete_btn.show()

        self.text_preview.setHtml("<center><h3>Reconstructing document from vector space...</h3></center>")
        
        # Pull chunks from LanceDB and stitch them together
        if self.store:
            content = self.store.reconstruct_document(self.active_filename)
            self.text_preview.setPlainText(content)
        else:
            self.text_preview.setPlainText("Error: Vector store not connected.")

    def _delete_active_document(self):
        if not self.active_filename: return

        reply = QMessageBox.question(
            self, 'Confirm Deletion',
            f"Are you sure you want to permanently delete and un-index '{self.active_filename}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info(f"User initiated deletion of {self.active_filename}")
            
            # 1. Erase from Vector Store (LanceDB)
            if self.store:
                self.store.delete_document(self.active_filename)
            
            # 2. Erase from Metadata Registry (SQLite)
            self.db.delete_document_metadata(self.active_filename)
            
            # 3. Reset UI
            self.active_filename = None
            self.doc_title.setText("No Document Selected")
            self.doc_stats.setText("")
            self.delete_btn.hide()
            self.text_preview.setHtml("<center><h3>Document deleted.</h3></center>")
            
            self.refresh_library_list()

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
            
            reply = QMessageBox.question(
                self, 'Overwrite Files?', msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Purge the old files from both LanceDB and SQLite before continuing
                logger.info("User chose to overwrite existing files. Purging old data...")
                for name in existing_files:
                    if self.store: self.store.delete_document(name)
                    self.db.delete_document_metadata(name)
            else:
                # If they say No, filter the existing files out of the ingestion list
                paths = [p for p in paths if p.name not in existing_files]
                if not paths:
                    logger.info("Ingestion cancelled; all selected files were duplicates and user declined overwrite.")
                    return # Exit early if nothing is left to ingest

        # 3. Proceed with the standard ingestion UI updates
        self.ingest_progress.setValue(0)
        self.ingest_progress.show()
        self.add_btn.setEnabled(False)
        self.add_btn.setText(" Ingesting...")
        
        logger.info(f"Emitting {len(paths)} files to main pipeline for ingestion.")
        self.ingest_requested.emit(paths)

    # --- UI Receivers for Worker Progress ---
    def update_ingestion_progress(self, percent: int):
        self.ingest_progress.setValue(percent)

    def show_error(self, error_msg: str):
        """Displays ingestion errors to the user and resets the UI."""
        QMessageBox.warning(self, "Ingestion Warning", error_msg)
        self.ingest_progress.hide()
        self.ingest_progress.setValue(0) # Force visual reset
        self.add_btn.setEnabled(True)
        self.add_btn.setText(" Ingest New Document")

    def complete_ingestion(self, total_chunks: int):
        self.ingest_progress.hide()
        self.ingest_progress.setValue(0) # Force visual reset
        self.add_btn.setEnabled(True)
        self.add_btn.setText(" Ingest New Document")
        self.refresh_library_list() # Instantly show the new files!
        
        # Explicitly tell the user if the document was a blank/scanned PDF
        if total_chunks == 0:
            QMessageBox.information(
                self, 
                "Ingestion Complete", 
                "Process finished, but 0 chunks were added. This usually means the file was already in the database, or it is a scanned PDF with no readable text."
            )