"""Tests for first-run bootstrap manifest and selection persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.bootstrap_manifest import (
    ADVANCED_ORDER,
    BOOTSTRAP_MODELS,
    BootstrapModelId,
    BootstrapModelTier,
    CONSENT_HIDDEN_MODEL_IDS,
    RECOMMENDED_ORDER,
    bootstrap_model_tier,
    bootstrap_tier_tag,
    consent_model_order,
    consent_tier_tag,
    default_selection,
    format_bootstrap_tier_tag_tooltip,
    format_byte_size,
    normalize_selection,
    OPTIONAL_RECOMMENDED_IDS,
    total_selected_bytes,
)
from core.bootstrap_selection import (
    KEY_COMPLETED,
    KEY_SELECTED,
    KEY_VOICE_IN,
    KEY_VOICE_OUT,
    _deserialize_selected,
    _serialize_selected,
    get_selected_model_ids,
    get_voice_input_default,
    get_voice_output_default,
    is_bootstrap_completed,
    maybe_seed_bootstrap_selection_for_existing_install,
    save_bootstrap_selection,
    should_show_bootstrap_consent,
)


def test_recommended_defaults_include_locked_core():
    selected = default_selection(advanced=False)
    assert BootstrapModelId.SIDECAR_QWEN17 in selected
    assert BootstrapModelId.WHISPER_SMALL in selected
    assert BootstrapModelId.KOKORO_TTS in selected
    assert BootstrapModelId.SEARCH_PRESET_BALANCED in selected
    assert BootstrapModelId.LLM_QWEN35_9B in selected
    assert BootstrapModelId.LLM_GEMMA4_E4B not in selected


def test_bootstrap_model_tiers():
    assert bootstrap_model_tier(BootstrapModelId.SIDECAR_QWEN17) is BootstrapModelTier.REQUIRED
    assert bootstrap_model_tier(BootstrapModelId.SEARCH_PRESET_BALANCED) is BootstrapModelTier.REQUIRED
    assert bootstrap_model_tier(BootstrapModelId.WHISPER_SMALL) is BootstrapModelTier.RECOMMENDED
    assert bootstrap_model_tier(BootstrapModelId.LLM_QWEN35_9B) is BootstrapModelTier.RECOMMENDED
    assert bootstrap_model_tier(BootstrapModelId.SIDECAR_QWEN05) is BootstrapModelTier.OPTIONAL
    assert bootstrap_model_tier(BootstrapModelId.LLM_GEMMA4_E4B) is BootstrapModelTier.OPTIONAL


def test_bootstrap_tier_tag_labels():
    assert bootstrap_tier_tag(BootstrapModelId.SIDECAR_QWEN17) == (
        "Required",
        "BootstrapTierTagRequired",
    )
    assert bootstrap_tier_tag(BootstrapModelId.KOKORO_TTS) == (
        "Recommended",
        "BootstrapTierTagRecommended",
    )
    assert bootstrap_tier_tag(BootstrapModelId.LLM_NEMOTRON_NANO) == (
        "Optional",
        "BootstrapTierTagOptional",
    )


def test_bootstrap_tier_tag_tooltips():
    required = format_bootstrap_tier_tag_tooltip(BootstrapModelId.SIDECAR_QWEN17)
    assert "Required" in required
    assert "Technical:" in required
    recommended = format_bootstrap_tier_tag_tooltip(BootstrapModelId.WHISPER_SMALL)
    assert "Recommended" in recommended
    optional = format_bootstrap_tier_tag_tooltip(BootstrapModelId.SIDECAR_QWEN05)
    assert "Optional" in optional
    assert "alternative" in optional.lower()


def test_advanced_defaults_include_core_and_balanced_search():
    selected = default_selection(advanced=True)
    assert selected == {
        BootstrapModelId.SIDECAR_QWEN17,
        BootstrapModelId.SEARCH_PRESET_BALANCED,
    }


def test_normalize_selection_mutual_exclusion():
    both_sidecars = {
        BootstrapModelId.SIDECAR_QWEN17,
        BootstrapModelId.SIDECAR_QWEN05,
    }
    normalized = normalize_selection(both_sidecars)
    assert BootstrapModelId.SIDECAR_QWEN17 in normalized
    assert BootstrapModelId.SIDECAR_QWEN05 not in normalized

    both_llms = {
        BootstrapModelId.SIDECAR_QWEN17,
        BootstrapModelId.LLM_QWEN35_9B,
        BootstrapModelId.LLM_GEMMA4_E4B,
    }
    normalized_llm = normalize_selection(both_llms)
    assert BootstrapModelId.LLM_QWEN35_9B in normalized_llm
    assert BootstrapModelId.LLM_GEMMA4_E4B not in normalized_llm

    all_llms = {
        BootstrapModelId.SIDECAR_QWEN17,
        BootstrapModelId.LLM_QWEN35_9B,
        BootstrapModelId.LLM_GEMMA4_E4B,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    }
    normalized_all_llms = normalize_selection(all_llms)
    assert BootstrapModelId.LLM_QWEN35_9B in normalized_all_llms
    assert normalized_all_llms & {
        BootstrapModelId.LLM_GEMMA4_E4B,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    } == set()


def test_total_selected_bytes_matches_catalog():
    from core.bootstrap_selection import total_selected_bytes as selection_total

    selected = default_selection(advanced=False)
    expected = sum(BOOTSTRAP_MODELS[mid].size_bytes for mid in selected)
    assert total_selected_bytes(selected) == expected
    assert selection_total(selected) == expected
    assert format_byte_size(expected)


def test_total_selected_bytes_uses_dynamic_sizes():
    from core.bootstrap_selection import total_selected_bytes as selection_total

    selected = {BootstrapModelId.SIDECAR_QWEN17}
    sizes = {BootstrapModelId.SIDECAR_QWEN17: 200}
    assert selection_total(selected, sizes=sizes) == 200


def test_balanced_search_in_recommended_order():
    assert BootstrapModelId.SEARCH_PRESET_BALANCED in RECOMMENDED_ORDER
    assert BootstrapModelId.SEARCH_PRESET_BALANCED in ADVANCED_ORDER
    assert RECOMMENDED_ORDER.index(BootstrapModelId.SEARCH_PRESET_BALANCED) == 1
    assert ADVANCED_ORDER.index(BootstrapModelId.SEARCH_PRESET_BALANCED) == 1


def test_consent_order_hides_deferred_sidecar():
    assert BootstrapModelId.SIDECAR_QWEN05 in CONSENT_HIDDEN_MODEL_IDS
    assert BootstrapModelId.SIDECAR_QWEN05 not in consent_model_order(advanced=True)
    assert BootstrapModelId.SIDECAR_QWEN05 not in consent_model_order(advanced=False)


def test_consent_tier_tag_advanced_core_models():
    assert consent_tier_tag(BootstrapModelId.SIDECAR_QWEN17, advanced=True) == (
        "Strongly Recommended",
        "BootstrapTierTagStronglyRecommended",
    )
    assert consent_tier_tag(BootstrapModelId.SEARCH_PRESET_BALANCED, advanced=True) == (
        "Strongly Recommended",
        "BootstrapTierTagStronglyRecommended",
    )
    assert consent_tier_tag(BootstrapModelId.SIDECAR_QWEN17, advanced=False) == bootstrap_tier_tag(
        BootstrapModelId.SIDECAR_QWEN17
    )


def test_selection_serialization_roundtrip():
    selected = default_selection(advanced=False)
    raw = _serialize_selected(selected)
    assert json.loads(raw)
    assert _deserialize_selected(raw) == selected


def test_optional_recommended_ids():
    assert OPTIONAL_RECOMMENDED_IDS == {
        BootstrapModelId.WHISPER_SMALL,
        BootstrapModelId.KOKORO_TTS,
        BootstrapModelId.LLM_QWEN35_9B,
    }


def test_selection_within_budget_accounts_for_safety_buffer():
    from core.bootstrap_selection import (
        budget_headroom_bytes,
        can_add_model,
        required_bytes_for,
        selection_within_budget,
    )

    selected = default_selection(advanced=False)
    required = required_bytes_for(selected)
    headroom = budget_headroom_bytes(selected)
    assert required == total_selected_bytes(selected) + 500 * 1024 * 1024
    assert selection_within_budget(selected) == (headroom >= 0)
    if headroom > BOOTSTRAP_MODELS[BootstrapModelId.LLM_NEMOTRON_NANO].size_bytes:
        assert can_add_model(selected, BootstrapModelId.LLM_NEMOTRON_NANO)


def test_seed_existing_install_does_not_skip_consent(monkeypatch):
    import tempfile

    from core.settings_store import SettingsStore

    schema_path = Path(__file__).resolve().parent.parent / "assets" / "config" / "settings.schema.json"
    inferred = {BootstrapModelId.SIDECAR_QWEN17}
    with tempfile.TemporaryDirectory() as tmp:
        store = SettingsStore(user_path=Path(tmp) / "settings.json", schema_path=schema_path)
        monkeypatch.setattr(
            "core.bootstrap_selection.get_settings_store",
            lambda: store,
        )
        monkeypatch.setattr(
            "core.bootstrap_download.infer_installed_selection",
            lambda: inferred,
        )

        maybe_seed_bootstrap_selection_for_existing_install()

        assert store.get(KEY_COMPLETED) is not True
        assert get_selected_model_ids() == inferred
        assert should_show_bootstrap_consent() is True


def test_sidecar_qwen17_uses_unsloth_gguf_repo():
    spec = BOOTSTRAP_MODELS[BootstrapModelId.SIDECAR_QWEN17]
    assert spec.hf_repo == "unsloth/Qwen3-1.7B-GGUF"
    assert spec.hf_filename == "Qwen3-1.7B-Q6_K.gguf"


def test_nemotron_nano_uses_public_unsloth_gguf_repo():
    spec = BOOTSTRAP_MODELS[BootstrapModelId.LLM_NEMOTRON_NANO]
    assert spec.hf_repo == "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF"
    assert spec.hf_filename == "NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"
    assert "BF16-GGUF" not in spec.hf_repo


def test_save_bootstrap_selection_persists_voice_defaults(monkeypatch):
    import tempfile

    from core.settings_store import SettingsStore

    schema_path = Path(__file__).resolve().parent.parent / "assets" / "config" / "settings.schema.json"
    with tempfile.TemporaryDirectory() as tmp:
        store = SettingsStore(user_path=Path(tmp) / "settings.json", schema_path=schema_path)
        monkeypatch.setattr(
            "core.bootstrap_selection.get_settings_store",
            lambda: store,
        )
        monkeypatch.setattr("core.bootstrap_selection.apply_bootstrap_selection", lambda _s: None)

        voice_only = {BootstrapModelId.SIDECAR_QWEN17}
        save_bootstrap_selection(voice_only)

        assert store.get(KEY_COMPLETED) is True
        assert get_selected_model_ids() == voice_only
        assert get_voice_input_default() is False
        assert get_voice_output_default() is False

        with_voice = default_selection(advanced=False)
        save_bootstrap_selection(with_voice)
        assert get_voice_input_default() is True
        assert get_voice_output_default() is True
        assert store.get(KEY_SELECTED) == _serialize_selected(with_voice)
