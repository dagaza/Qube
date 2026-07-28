"""Offline licensing and pack signature verification."""

from core.licensing.license_schema import (
    LICENSE_FILE_EXTENSION,
    LICENSE_SCHEMA_VERSION,
    LicenseDocument,
    LicenseError,
    license_signing_payload,
    parse_license_document,
)
from core.licensing.schema import (
    PackSignatureError,
    PackSigning,
    PackVerificationResult,
    SIGNING_FIELD,
)
from core.licensing.sign import attach_signing_block, sign_payload_bytes
from core.licensing.store import (
    LicenseImportResult,
    format_license_status_text,
    import_license_from_path,
    get_active_license,
    license_summary,
    remove_license,
    set_license_cache_path,
)
from core.licensing.verify import (
    knowledge_pack_signing_payload,
    theme_pack_signing_payload,
    verify_knowledge_pack_signature,
    verify_license_document,
    verify_theme_pack_signature,
)

__all__ = [
    "LICENSE_FILE_EXTENSION",
    "LICENSE_SCHEMA_VERSION",
    "LicenseDocument",
    "LicenseError",
    "LicenseImportResult",
    "SIGNING_FIELD",
    "PackSignatureError",
    "PackSigning",
    "PackVerificationResult",
    "attach_signing_block",
    "format_license_status_text",
    "get_active_license",
    "import_license_from_path",
    "knowledge_pack_signing_payload",
    "license_signing_payload",
    "license_summary",
    "parse_license_document",
    "remove_license",
    "set_license_cache_path",
    "sign_payload_bytes",
    "theme_pack_signing_payload",
    "verify_knowledge_pack_signature",
    "verify_license_document",
    "verify_theme_pack_signature",
]
