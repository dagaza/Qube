"""Provider credential rows (Settings → Knowledge)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.knowledge.credentials import (
    connection_mode_display,
    env_override_active,
    resolve_credential,
)
from core.knowledge.provider_credentials import (
    ProviderCredentialSpec,
    get_provider_credential_spec,
    list_active_provider_credential_specs,
    provider_has_implemented_adapter,
)
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.views.settings.widgets import (
    add_subsection_to_layout,
    make_settings_hint,
    wrap_subsection,
)


def _build_provider_credential_card(host, spec: ProviderCredentialSpec) -> QWidget:
    """One bordered card per knowledge provider."""
    implemented = provider_has_implemented_adapter(spec)

    card = QWidget()
    card.setObjectName("SettingsLogCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(14, 12, 14, 12)
    card_layout.setSpacing(10)

    title = QLabel(spec.label)
    title.setObjectName("SettingsLogTitle")
    card_layout.addWidget(title)

    if spec.key_benefits:
        benefit_lbl = QLabel(spec.key_benefits)
        benefit_lbl.setWordWrap(True)
        benefit_lbl.setObjectName("SettingsLogDescription")
        card_layout.addWidget(benefit_lbl)

    form = QFormLayout()
    form.setSpacing(10)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

    key_field = QLineEdit()
    key_field.setEchoMode(QLineEdit.EchoMode.Password)
    key_field.setPlaceholderText(
        "Paste API key"
        if implemented or not spec.key_required
        else "Required when available"
    )
    key_field.setEnabled(implemented and (not spec.key_required or spec.supports_free_api_key))
    if spec.key_required and not implemented:
        key_field.setEnabled(False)
    key_field.setMinimumWidth(220)
    form.addRow("API key", key_field)
    card_layout.addLayout(form)

    test_btn = QPushButton("Test connection")
    test_btn.setToolTip("Verify connectivity with the current key (saved or typed).")
    test_btn.setEnabled(bool(spec.test_probe) and implemented)
    test_btn.clicked.connect(
        lambda _checked=False, pid=spec.provider_id, h=host: h._on_provider_credential_test(pid)
    )
    apply_brand_primary(test_btn, icon_name="fa5s.plug")

    signup_btn = QPushButton("Get free key")
    signup_btn.setToolTip("Open the provider sign-up page in your browser.")
    signup_btn.setEnabled(bool(spec.signup_url))
    signup_btn.clicked.connect(
        lambda _checked=False, pid=spec.provider_id, h=host: h._on_provider_credential_signup(
            pid
        )
    )

    clear_btn = QPushButton("Clear saved key")
    clear_btn.setToolTip("Remove the locally stored key for this provider.")
    clear_btn.clicked.connect(
        lambda _checked=False, pid=spec.provider_id, h=host: h._on_provider_credential_clear(pid)
    )
    apply_brand_danger(clear_btn, icon_name="fa5s.eraser")

    btn_row = QWidget()
    btn_row_layout = QHBoxLayout(btn_row)
    btn_row_layout.setContentsMargins(0, 0, 0, 0)
    btn_row_layout.setSpacing(8)
    btn_row_layout.addWidget(test_btn)
    btn_row_layout.addWidget(signup_btn)
    btn_row_layout.addWidget(clear_btn)
    btn_row_layout.addStretch(1)
    card_layout.addWidget(btn_row)

    status_lbl = QLabel()
    status_lbl.setWordWrap(True)
    status_lbl.setObjectName("SettingsLogStatus")
    card_layout.addWidget(status_lbl)

    if not spec.supports_anonymous and spec.key_required and not implemented:
        note = QLabel(
            "Retrieval stays disabled until this source is implemented and a key is saved."
        )
        note.setWordWrap(True)
        note.setObjectName("SettingsLogNote")
        card_layout.addWidget(note)

    key_field.editingFinished.connect(
        lambda pid=spec.provider_id, h=host: h._on_provider_credential_editing_finished(pid)
    )

    host.knowledge_provider_key_fields[spec.provider_id] = key_field
    host.knowledge_provider_status_labels[spec.provider_id] = status_lbl
    return card


def build_knowledge_provider_credentials_section(host) -> QWidget:
    """Build one credential card per knowledge provider id."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    host.knowledge_provider_key_fields: dict[str, QLineEdit] = {}
    host.knowledge_provider_status_labels: dict[str, QLabel] = {}

    add_subsection_to_layout(layout, "Provider credentials", anchor="knowledge_provider_credentials")

    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)
    inner_layout.setSpacing(12)

    intro = make_settings_hint(
        "Configure API keys for live retrieval providers. Qube works without keys "
        "wherever providers allow anonymous access; optional free keys improve "
        "quotas and reliability. Keys are stored locally on this device."
    )
    inner_layout.addWidget(intro)

    for spec in list_active_provider_credential_specs():
        inner_layout.addWidget(_build_provider_credential_card(host, spec))

    layout.addWidget(wrap_subsection(inner, anchor="knowledge_provider_credentials"))
    sync_provider_credential_rows(host)
    return container


def sync_provider_credential_rows(host) -> None:
    """Refresh status labels and field enablement from persisted credentials."""
    if not hasattr(host, "knowledge_provider_key_fields"):
        return

    for provider_id, key_field in host.knowledge_provider_key_fields.items():
        spec = get_provider_credential_spec(provider_id)
        if spec is None:
            continue

        env_locked = env_override_active(provider_id)
        cred = resolve_credential(provider_id)
        implemented = provider_has_implemented_adapter(spec)

        if env_locked:
            key_field.setEnabled(False)
            key_field.setPlaceholderText("Using environment variable override")
            key_field.clear()
        elif spec.key_required and not implemented:
            key_field.setEnabled(False)
        else:
            key_field.setEnabled(True)
            key_field.setPlaceholderText(
                "Key saved — enter new value to replace"
                if cred.mode.value == "user_key"
                else "Paste API key (optional)"
            )

        status = host.knowledge_provider_status_labels.get(provider_id)
        if status is not None:
            status.setText(f"Current mode: {connection_mode_display(provider_id)}")
