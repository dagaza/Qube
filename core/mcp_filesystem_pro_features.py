"""Pro MCP Filesystem integration — license + configured-source helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

PRO_MCP_FILESYSTEM_CAPABILITY = "pro.mcp_filesystem"
PRO_MCP_FILESYSTEM_FEATURE = "integrations.mcp_filesystem"

FILESYSTEM_MCP_NAMESPACE = "filesystem"

LICENSE_REQUIRED_MESSAGE = (
    "MCP Filesystem integration requires a Qube Pro (or Team) license.\n\n"
    "Import your license under Settings → License."
)

if TYPE_CHECKING:
    from core.integrations.mcp_configured_source import McpConfiguredBinding
    from core.knowledge.configured_sources import ConfiguredSource


def user_has_pro_mcp_filesystem() -> bool:
    from core.capabilities import has_feature

    return has_feature(PRO_MCP_FILESYSTEM_FEATURE)


def is_mcp_filesystem_namespace(namespace: str) -> bool:
    return (namespace or "").strip().lower() == FILESYSTEM_MCP_NAMESPACE


def is_mcp_filesystem_config(config: dict[str, Any], *, namespace: str = "") -> bool:
    cfg = dict(config or {})
    ns = (namespace or cfg.get("namespace") or cfg.get("adapter_id") or "").strip().lower()
    if is_mcp_filesystem_namespace(ns):
        return True
    command = cfg.get("command")
    if not isinstance(command, list) or not command:
        return False
    joined = " ".join(str(part).lower() for part in command)
    return (
        "server-filesystem" in joined
        or "mcp-server-filesystem" in joined
        or "/server-filesystem" in joined
    )


def is_mcp_filesystem_source(source: ConfiguredSource) -> bool:
    if str(source.connector_type or "").strip().lower() != "mcp":
        return False
    cfg = dict(source.config or {})
    ns = str(cfg.get("namespace") or cfg.get("adapter_id") or source.id)
    return is_mcp_filesystem_config(cfg, namespace=ns)


def is_mcp_filesystem_binding(binding: McpConfiguredBinding) -> bool:
    if is_mcp_filesystem_namespace(binding.namespace):
        return True
    return is_mcp_filesystem_config(
        {"command": list(binding.command), "namespace": binding.namespace},
        namespace=binding.namespace,
    )


def mcp_filesystem_integration_allowed(
    *,
    config: dict[str, Any] | None = None,
    namespace: str = "",
    source: ConfiguredSource | None = None,
    binding: McpConfiguredBinding | None = None,
) -> bool:
    if user_has_pro_mcp_filesystem():
        return True
    if source is not None and is_mcp_filesystem_source(source):
        return False
    if binding is not None and is_mcp_filesystem_binding(binding):
        return False
    if config is not None and is_mcp_filesystem_config(config, namespace=namespace):
        return False
    if namespace and is_mcp_filesystem_namespace(namespace):
        return False
    return True


def require_pro_mcp_filesystem_for_source(source: ConfiguredSource) -> None:
    if is_mcp_filesystem_source(source) and not user_has_pro_mcp_filesystem():
        raise ValueError(LICENSE_REQUIRED_MESSAGE)


def require_pro_mcp_filesystem_for_config(
    config: dict[str, Any],
    *,
    namespace: str,
) -> None:
    if is_mcp_filesystem_config(config, namespace=namespace) and not user_has_pro_mcp_filesystem():
        raise ValueError(LICENSE_REQUIRED_MESSAGE)


def require_pro_mcp_filesystem_for_namespace(namespace: str) -> None:
    if is_mcp_filesystem_namespace(namespace) and not user_has_pro_mcp_filesystem():
        raise ValueError(LICENSE_REQUIRED_MESSAGE)


def require_pro_mcp_filesystem() -> None:
    from core.capabilities import require_feature

    require_feature(PRO_MCP_FILESYSTEM_FEATURE)


def sync_mcp_filesystem_pro_features(host) -> bool:
    """Refresh MCP integration UI after license changes."""
    from core.integrations.descriptor_cache import reconcile_mcp_integration_state

    summary = reconcile_mcp_integration_state()
    changed = any(summary.values())
    from ui.views.settings.sections.knowledge_custom_sources import (
        _refresh_integrations_consent_if_available,
    )

    _refresh_integrations_consent_if_available(host)
    if hasattr(host, "integrations_mcp_servers_layout"):
        from ui.views.settings.sections.integrations import sync_integrations_mcp_servers_panel

        is_dark = getattr(host.window(), "_is_dark_theme", True)
        sync_integrations_mcp_servers_panel(host, is_dark=is_dark)
    return changed
