"""AI & Models settings section — engine, local GGUF, generation, cognition, startup, chat."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import (
    get_advanced_chat_template_unlocked,
    get_advanced_engine_unlocked,
    get_advanced_hardware_unlocked,
    get_auto_load_last_model_on_startup,
    get_enable_chat_personality_nudge,
    get_internal_n_gpu_layers,
    get_internal_n_threads,
    get_llm_chat_history_messages,
    get_llm_context_limit,
    get_llm_min_p,
    get_llm_output_token_limit,
    get_llm_output_token_limit_enabled,
    get_llm_presence_penalty,
    get_llm_repeat_penalty,
    get_llm_temperature,
    get_llm_top_k,
    get_llm_top_p,
    get_skills_enabled,
    set_auto_load_last_model_on_startup,
)
from core.auxiliary_cognition import get_cognition_models_dir
from core.cpu_threads import max_cpu_threads_for_ui
from core.gpu_layers_cap import max_safe_n_gpu_layers, is_unified_gpu_memory
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle
from ui.views.settings.controls import (
    NoScrollDoubleSpinBox,
    NoScrollSlider,
    NoScrollSpinBox,
    SettingsScrollListWidget,
)
from ui.views.settings.widgets import (
    add_subsection_to_form,
    add_section_reset_footer,
    make_disclosure_row,
    make_external_engine_hint,
    track_internal_ai_label,
    wrap_subsection,
)


def build_section(host, *, is_dark: bool) -> QWidget:
    ai_widget = QWidget()
    ai_widget.setObjectName("SettingsFormContainer")
    ai_form = QFormLayout(ai_widget)
    ai_form.setSpacing(15)
    ai_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    # --- Engine ---
    add_subsection_to_form(ai_form, "Engine & routing", anchor="engine")

    host.engine_selector = SelectorButton("Select engine...", is_dark=is_dark)
    host.engine_selector.setObjectName("SettingsEngineSelector")
    host.provider_selector = SelectorButton("Select Provider...", is_dark=is_dark)

    for btn in (host.engine_selector, host.provider_selector):
        btn.setMaximumWidth(250)
        btn.setMenu(QMenu(btn))

    host.engine_selector.setToolTip(
        "Internal runs downloaded .gguf models on this device. "
        "External connects to LM Studio or Ollama."
    )
    host.provider_selector.setToolTip(
        "OpenAI-compatible server to use when External inference is selected."
    )

    ai_form.addRow("AI Engine", host.engine_selector)
    ai_form.addRow("External Provider", host.provider_selector)

    ai_form.addRow("", make_external_engine_hint(host))

    # --- Local models ---
    track_internal_ai_label(
        host, add_subsection_to_form(ai_form, "Local models", anchor="local_models")
    )

    local_models_inner = QWidget()
    local_models_form = QFormLayout(local_models_inner)
    local_models_form.setSpacing(15)
    local_models_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.models_dir_label = QLabel()
    host.models_dir_label.setWordWrap(True)

    local_row = QHBoxLayout()
    host.local_gguf_list = SettingsScrollListWidget()
    host.local_gguf_list.setMinimumHeight(100)
    host.local_gguf_list.setMaximumHeight(160)
    host.local_gguf_list.setToolTip(
        "Downloaded .gguf models on this device. Select one, then click Use selected."
    )
    local_row.addWidget(host.local_gguf_list, stretch=1)
    local_btn_col = QVBoxLayout()
    local_btn_col.setSpacing(8)
    host.use_local_gguf_btn = QPushButton("Use selected")
    apply_brand_primary(host.use_local_gguf_btn)
    host.use_local_gguf_btn.setToolTip("Activate a downloaded .gguf for the native engine")
    host.use_local_gguf_btn.clicked.connect(host._apply_selected_local_gguf)
    local_btn_col.addWidget(host.use_local_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.refresh_local_gguf_btn = QPushButton("Refresh")
    host.refresh_local_gguf_btn.setToolTip(
        "Rescan the models folder for .gguf files added while the app is running"
    )
    host.refresh_local_gguf_btn.clicked.connect(host._on_refresh_local_gguf_clicked)
    local_btn_col.addWidget(host.refresh_local_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop)
    host.delete_local_gguf_btn = QPushButton("Delete")
    apply_brand_danger(host.delete_local_gguf_btn)
    host.delete_local_gguf_btn.setToolTip("Permanently delete the selected .gguf file from disk")
    host.delete_local_gguf_btn.clicked.connect(host._delete_selected_local_gguf)
    local_btn_col.addWidget(host.delete_local_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop)
    local_row.addLayout(local_btn_col)

    host.active_native_model_lbl = QLabel()

    local_models_form.addRow("Model storage", host.models_dir_label)
    local_models_form.addRow("On this device", local_row)
    local_models_form.addRow("Active model", host.active_native_model_lbl)

    host._ai_local_models_subsection = wrap_subsection(
        local_models_inner, anchor="local_models"
    )
    ai_form.addRow(host._ai_local_models_subsection)

    # --- Startup ---
    track_internal_ai_label(
        host, add_subsection_to_form(ai_form, "Startup", anchor="startup")
    )

    startup_inner = QWidget()
    startup_form = QFormLayout(startup_inner)
    startup_form.setSpacing(15)
    startup_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.auto_load_last_model_cb = QCheckBox("Load last used model on startup")
    host.auto_load_last_model_cb.setToolTip(
        "Automatically loads the last used model at startup. This may significantly "
        "increase application startup time depending on the model size and your hardware."
    )
    host.auto_load_last_model_cb.setChecked(get_auto_load_last_model_on_startup())
    host.auto_load_last_model_cb.toggled.connect(set_auto_load_last_model_on_startup)
    host.auto_load_last_model_cb.toggled.connect(host.auto_load_last_model_changed.emit)
    startup_form.addRow("", host.auto_load_last_model_cb)

    host._ai_startup_subsection = wrap_subsection(startup_inner, anchor="startup")
    ai_form.addRow(host._ai_startup_subsection)

    # --- Generation ---
    add_subsection_to_form(ai_form, "Generation", anchor="generation")

    host._generation_spinboxes: list = []

    _gen_temp_tip = (
        "Creativity Slider: Lower values (0.1-0.3) produce strict, factual answers,  "
        "but will make the answers sound more robotic and less natural. "
        "Higher values (0.7-1.0) make Qube more creative and will make the answers sound more natural. "
        "It is recommended to keep the temperature around 0.7 - 0.8 for balanced performance."
    )
    _gen_ctx_tip = (
        "Total token budget for one turn: instructions, retrieved documents, "
        "chat history, your message, and the reply all share this window. "
        "On the local GGUF engine this sets n_ctx and reloads the model when "
        "changed; higher values use more RAM/VRAM. Reply length is capped "
        "separately below — both settings draw from the same pool."
    )
    _gen_output_limit_tip = (
        "When enabled, each reply stops after the max reply tokens you set "
        "(the model's max_tokens). When disabled, the reply may grow until the "
        "context window is full minus whatever the prompt already used. Chat "
        "history (toolbar) and retrieval consume prompt space, so long threads "
        "can shorten replies even with this off."
    )
    _gen_output_tokens_tip = (
        "Upper bound on new tokens per assistant reply when the limit above is "
        "on. This does not add extra capacity — prompt tokens (chat history, "
        "RAG, system text) are counted first inside the context window. For "
        "very long answers, turn the limit off or lower chat history in the toolbar."
    )
    _gen_history_tip = (
        "How many recent user/assistant messages to include in each prompt. "
        "More history improves continuity but uses more of the context window "
        "for the prompt, leaving less room for long replies. Also increases "
        "RAM/VRAM during inference. Long-term memory still covers facts dropped "
        "from this sliding window."
    )
    _gen_top_k_tip = (
        "Top-K sampling: only the K most likely next tokens are considered. "
        "0 disables top-K filtering."
    )
    _gen_repeat_tip = (
        "Repeat penalty: values above 1.0 discourage the model from repeating recent "
        "words or phrases."
    )
    _gen_presence_tip = (
        "Presence penalty: discourages tokens that have already appeared anywhere in "
        "the current output."
    )
    _gen_top_p_tip = (
        "Top-P (nucleus) sampling: keeps the smallest set of tokens whose cumulative "
        "probability reaches P."
    )
    _gen_min_p_tip = (
        "Min-P sampling: drops tokens below this relative probability floor. "
        "0 disables min-P filtering."
    )
    _gen_advanced_tip = (
        "Optional sampling controls for power users. Defaults work well for most models."
    )

    host.llm_temp_spin = NoScrollDoubleSpinBox()
    host.llm_temp_spin.setRange(0.0, 2.0)
    host.llm_temp_spin.setSingleStep(0.1)
    host.llm_temp_spin.setValue(get_llm_temperature())
    host._add_generation_form_row(ai_form, "Temperature", _gen_temp_tip, host.llm_temp_spin)

    host.llm_ctx_spin = NoScrollSpinBox()
    host.llm_ctx_spin.setRange(1024, 128000)
    host.llm_ctx_spin.setSingleStep(256)
    host.llm_ctx_spin.setValue(get_llm_context_limit())
    host._add_generation_form_row(ai_form, "Context limit", _gen_ctx_tip, host.llm_ctx_spin)

    host.llm_output_limit_cb = QCheckBox("Limit maximum reply length")
    host.llm_output_limit_cb.setChecked(get_llm_output_token_limit_enabled())
    host.llm_output_limit_cb.setToolTip(_gen_output_limit_tip)
    output_limit_row = QWidget()
    output_limit_layout = QHBoxLayout(output_limit_row)
    output_limit_layout.setContentsMargins(0, 0, 0, 0)
    output_limit_layout.setSpacing(6)
    output_limit_layout.addWidget(host.llm_output_limit_cb)
    output_limit_layout.addWidget(host._make_settings_info_button(_gen_output_limit_tip))
    output_limit_layout.addStretch(1)
    ai_form.addRow("", output_limit_row)

    host.llm_output_limit_spin = NoScrollSpinBox()
    host.llm_output_limit_spin.setRange(256, 32768)
    host.llm_output_limit_spin.setSingleStep(256)
    host.llm_output_limit_spin.setValue(get_llm_output_token_limit())
    host._add_generation_form_row(
        ai_form,
        "Max reply tokens",
        _gen_output_tokens_tip,
        host.llm_output_limit_spin,
    )

    host.llm_output_limit_hint = QLabel()
    host.llm_output_limit_hint.setWordWrap(True)
    host.llm_output_limit_hint.setProperty("class", "SettingsHint")
    ai_form.addRow("", host.llm_output_limit_hint)
    host._refresh_output_token_limit_hint()

    host.llm_history_spin = NoScrollSpinBox()
    host.llm_history_spin.setRange(2, 100)
    host.llm_history_spin.setSingleStep(2)
    host.llm_history_spin.setValue(get_llm_chat_history_messages())
    host._add_generation_form_row(
        ai_form, "Chat history", _gen_history_tip, host.llm_history_spin
    )
    host.llm_history_spin.valueChanged.connect(
        lambda _v: host._refresh_output_token_limit_hint()
    )

    host.generation_advanced_toggle, gen_adv_row, host.generation_advanced_panel = (
        make_disclosure_row(
            host,
            "Show advanced generation settings",
            _gen_advanced_tip,
        )
    )
    host.generation_advanced_toggle.blockSignals(True)
    host.generation_advanced_toggle.setChecked(False)
    host.generation_advanced_toggle.blockSignals(False)
    host.generation_advanced_panel.setVisible(False)
    host.generation_advanced_toggle.toggled.connect(
        host.generation_advanced_panel.setVisible
    )
    ai_form.addRow("", gen_adv_row)

    gen_adv_form_widget = QWidget()
    gen_adv_form = QFormLayout(gen_adv_form_widget)
    gen_adv_form.setSpacing(15)
    gen_adv_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.llm_top_k_spin = NoScrollSpinBox()
    host.llm_top_k_spin.setRange(0, 200)
    host.llm_top_k_spin.setValue(get_llm_top_k())
    host._add_generation_form_row(
        gen_adv_form, "Top-K sampling", _gen_top_k_tip, host.llm_top_k_spin
    )

    host.llm_top_p_spin = NoScrollDoubleSpinBox()
    host.llm_top_p_spin.setRange(0.0, 1.0)
    host.llm_top_p_spin.setSingleStep(0.01)
    host.llm_top_p_spin.setValue(get_llm_top_p())
    host._add_generation_form_row(
        gen_adv_form, "Top-P sampling", _gen_top_p_tip, host.llm_top_p_spin
    )

    host.llm_min_p_spin = NoScrollDoubleSpinBox()
    host.llm_min_p_spin.setRange(0.0, 1.0)
    host.llm_min_p_spin.setSingleStep(0.01)
    host.llm_min_p_spin.setValue(get_llm_min_p())
    host._add_generation_form_row(
        gen_adv_form, "Min-P sampling", _gen_min_p_tip, host.llm_min_p_spin
    )

    host.llm_repeat_penalty_spin = NoScrollDoubleSpinBox()
    host.llm_repeat_penalty_spin.setRange(0.0, 2.0)
    host.llm_repeat_penalty_spin.setSingleStep(0.05)
    host.llm_repeat_penalty_spin.setValue(get_llm_repeat_penalty())
    host._add_generation_form_row(
        gen_adv_form, "Repeat penalty", _gen_repeat_tip, host.llm_repeat_penalty_spin
    )

    host.llm_presence_penalty_spin = NoScrollDoubleSpinBox()
    host.llm_presence_penalty_spin.setRange(0.0, 2.0)
    host.llm_presence_penalty_spin.setSingleStep(0.05)
    host.llm_presence_penalty_spin.setValue(get_llm_presence_penalty())
    host._add_generation_form_row(
        gen_adv_form,
        "Presence penalty",
        _gen_presence_tip,
        host.llm_presence_penalty_spin,
    )

    host.generation_advanced_panel.layout().addWidget(gen_adv_form_widget)
    ai_form.addRow("", host.generation_advanced_panel)

    # --- Chat style ---
    add_subsection_to_form(ai_form, "Chat style", anchor="chat_style")

    host.chat_personality_toggle = PrestigeToggle()
    host.chat_personality_label = QLabel("Encourage brief follow-ups on general chat")
    host.chat_personality_label.setWordWrap(True)
    _chat_personality_tip = (
        "When enabled, plain chat turns (no library or memory sources) "
        "gently invite one optional short follow-up—e.g. after a joke or "
        "story—not on retrieval, web search, or remember-this turns. "
        "On by default."
    )
    host.chat_personality_toggle.setToolTip(_chat_personality_tip)
    host.chat_personality_label.setToolTip(_chat_personality_tip)
    chat_personality_row = QWidget()
    chat_personality_row_layout = QHBoxLayout(chat_personality_row)
    chat_personality_row_layout.setContentsMargins(0, 0, 0, 0)
    chat_personality_row_layout.addWidget(
        host.chat_personality_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    chat_personality_row_layout.addWidget(host.chat_personality_label, stretch=1)
    host.chat_personality_toggle.blockSignals(True)
    host.chat_personality_toggle.setChecked(get_enable_chat_personality_nudge())
    host.chat_personality_toggle.blockSignals(False)
    host.chat_personality_toggle.toggled.connect(host._on_chat_personality_toggled)
    ai_form.addRow("", chat_personality_row)

    # --- Reasoning skills ---
    add_subsection_to_form(ai_form, "Reasoning skills", anchor="skills")

    host.skills_enabled_toggle = PrestigeToggle()
    host.skills_enabled_label = QLabel("Enable compositional reasoning skills")
    host.skills_enabled_label.setWordWrap(True)
    _skills_enabled_tip = (
        "When enabled, Qube auto-detects up to three reasoning skills per turn and "
        "injects non-authoritative prompt guidance after routing (does not change "
        "routes or tools). Type @ in the composer to force a specific skill with "
        "@[skill:skill_id] — forced skills work even when this toggle is off. "
        "Off by default."
    )
    host.skills_enabled_toggle.setToolTip(_skills_enabled_tip)
    host.skills_enabled_label.setToolTip(_skills_enabled_tip)
    skills_row = QWidget()
    skills_row_layout = QHBoxLayout(skills_row)
    skills_row_layout.setContentsMargins(0, 0, 0, 0)
    skills_row_layout.addWidget(
        host.skills_enabled_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    skills_row_layout.addWidget(host.skills_enabled_label, stretch=1)
    host.skills_enabled_toggle.blockSignals(True)
    host.skills_enabled_toggle.setChecked(get_skills_enabled())
    host.skills_enabled_toggle.blockSignals(False)
    host.skills_enabled_toggle.toggled.connect(host._on_skills_enabled_toggled)
    ai_form.addRow("", skills_row)

    # --- Hardware tuning ---
    track_internal_ai_label(
        host, add_subsection_to_form(ai_form, "Hardware tuning", anchor="hardware")
    )

    _hardware_adv_tip = (
        "Advanced hardware controls are not for everyday use.\n\n"
        "Unlocks GPU offload layers and CPU thread pool tuning for the native engine. "
        "Setting GPU layers too high can exhaust video memory and crash the app."
    )
    host.advanced_hardware_toggle, host.advanced_hardware_row, host.advanced_hardware_panel = (
        make_disclosure_row(
            host,
            "Show advanced hardware settings",
            _hardware_adv_tip,
        )
    )
    host.advanced_hardware_toggle.blockSignals(True)
    host.advanced_hardware_toggle.setChecked(get_advanced_hardware_unlocked())
    host.advanced_hardware_toggle.blockSignals(False)
    host.advanced_hardware_toggle.toggled.connect(host._on_advanced_hardware_toggled)
    ai_form.addRow("", host.advanced_hardware_row)

    hardware_inner = QWidget()
    hardware_form = QFormLayout(hardware_inner)
    hardware_form.setSpacing(15)
    hardware_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host._gpu_layers_cap = max_safe_n_gpu_layers()
    gpu_layers_row = QWidget()
    gpu_layers_row_layout = QHBoxLayout(gpu_layers_row)
    gpu_layers_row_layout.setContentsMargins(0, 0, 0, 0)
    gpu_layers_row_layout.setSpacing(12)

    host.gpu_layers_slider = NoScrollSlider(Qt.Orientation.Horizontal)
    host.gpu_layers_slider.setMinimum(0)
    host.gpu_layers_slider.setMaximum(host._gpu_layers_cap)
    host.gpu_layers_slider.setSingleStep(1)
    host.gpu_layers_slider.setPageStep(
        max(1, host._gpu_layers_cap // 10) if host._gpu_layers_cap else 1
    )
    _gpu_val = get_internal_n_gpu_layers()
    host.gpu_layers_slider.blockSignals(True)
    host.gpu_layers_slider.setValue(_gpu_val)
    host.gpu_layers_slider.blockSignals(False)

    host.gpu_layers_value_lbl = QLabel(str(_gpu_val))
    host.gpu_layers_value_lbl.setMinimumWidth(44)
    host.gpu_layers_value_lbl.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    _gpu_tip = (
        "The number of AI 'brain layers' loaded into your graphics card (GPU). "
        "More layers make the AI generate text much faster, but setting this too high "
        "may use up all your video memory and cause crashes."
    )
    if is_unified_gpu_memory():
        _gpu_tip += (
            " On unified-memory systems (Apple Silicon or AMD APUs), layers draw from "
            "shared system RAM — raise this toward the maximum for much better speed."
        )
    host.gpu_layers_slider.setToolTip(_gpu_tip)
    host.gpu_layers_value_lbl.setToolTip(_gpu_tip)
    gpu_layers_row.setToolTip(_gpu_tip)

    host.gpu_layers_slider.valueChanged.connect(host._on_gpu_layers_slider_changed)

    gpu_layers_row_layout.addWidget(host.gpu_layers_slider, stretch=1)
    gpu_layers_row_layout.addWidget(host.gpu_layers_value_lbl)

    host._cpu_threads_max = max_cpu_threads_for_ui()
    cpu_threads_row = QWidget()
    cpu_threads_row_layout = QHBoxLayout(cpu_threads_row)
    cpu_threads_row_layout.setContentsMargins(0, 0, 0, 0)
    cpu_threads_row_layout.setSpacing(12)

    host.cpu_threads_slider = NoScrollSlider(Qt.Orientation.Horizontal)
    host.cpu_threads_slider.setMinimum(1)
    host.cpu_threads_slider.setMaximum(host._cpu_threads_max)
    host.cpu_threads_slider.setSingleStep(1)
    host.cpu_threads_slider.setPageStep(max(1, host._cpu_threads_max // 10))
    _cpu_val = get_internal_n_threads()
    host.cpu_threads_slider.blockSignals(True)
    host.cpu_threads_slider.setValue(_cpu_val)
    host.cpu_threads_slider.blockSignals(False)

    host.cpu_threads_value_lbl = QLabel(str(_cpu_val))
    host.cpu_threads_value_lbl.setMinimumWidth(44)
    host.cpu_threads_value_lbl.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    _cpu_tip = (
        "How many processor cores the AI is allowed to use. Setting this close to your "
        "computer's total cores speeds up generation, but might slow down other "
        "applications running in the background."
    )
    host.cpu_threads_slider.setToolTip(_cpu_tip)
    host.cpu_threads_value_lbl.setToolTip(_cpu_tip)
    cpu_threads_row.setToolTip(_cpu_tip)

    host.cpu_threads_slider.valueChanged.connect(host._on_cpu_threads_slider_changed)

    cpu_threads_row_layout.addWidget(host.cpu_threads_slider, stretch=1)
    cpu_threads_row_layout.addWidget(host.cpu_threads_value_lbl)

    hardware_form.addRow("GPU offload layers", gpu_layers_row)
    hardware_form.addRow("CPU thread pool", cpu_threads_row)

    host._ai_hardware_subsection = wrap_subsection(hardware_inner, anchor="hardware")
    host.advanced_hardware_panel.layout().addWidget(host._ai_hardware_subsection)
    host.advanced_hardware_panel.setVisible(get_advanced_hardware_unlocked())
    ai_form.addRow("", host.advanced_hardware_panel)

    # --- Inference transparency (read-only) ---
    track_internal_ai_label(
        host,
        add_subsection_to_form(ai_form, "Inference stack", anchor="inference_stack"),
    )
    host.inference_transparency_lbl = QLabel("Loading inference stack details…")
    host.inference_transparency_lbl.setWordWrap(True)
    host.inference_transparency_lbl.setProperty("class", "ToolsPaneControl")
    host.inference_transparency_lbl.setToolTip(
        "Read-only summary of llama.cpp build backend, hardware detection (including AMD APU "
        "unified memory), requested GPU layer configuration, and embedder/sidecar compute paths."
    )
    ai_form.addRow("", host.inference_transparency_lbl)

    # --- Chat template ---
    track_internal_ai_label(
        host, add_subsection_to_form(ai_form, "Chat template", anchor="chat_template")
    )

    _chat_template_adv_tip = (
        "Advanced chat template controls are not for everyday use.\n\n"
        "Unlocks manual chat template selection for the native engine. Auto usually "
        "matches the loaded model; an incorrect template can cause hallucinations or "
        "the model talking to itself."
    )
    (
        host.advanced_chat_template_toggle,
        host.advanced_chat_template_row,
        host.advanced_chat_template_panel,
    ) = make_disclosure_row(
        host,
        "Show advanced chat template settings",
        _chat_template_adv_tip,
    )
    host.advanced_chat_template_toggle.blockSignals(True)
    host.advanced_chat_template_toggle.setChecked(get_advanced_chat_template_unlocked())
    host.advanced_chat_template_toggle.blockSignals(False)
    host.advanced_chat_template_toggle.toggled.connect(
        host._on_advanced_chat_template_toggled
    )
    ai_form.addRow("", host.advanced_chat_template_row)

    chat_template_inner = QWidget()
    chat_template_form = QFormLayout(chat_template_inner)
    chat_template_form.setSpacing(15)
    chat_template_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.native_chat_format_selector = SelectorButton(
        "Select chat template...", is_dark=is_dark
    )
    host.native_chat_format_selector.setMaximumWidth(350)
    host.native_chat_format_selector.setMenu(QMenu(host.native_chat_format_selector))
    host.native_chat_format_selector.setToolTip(
        "The specific conversational format this AI model was trained on. If the native "
        "engine is hallucinating or talking to itself, changing this to match the model's "
        "family (e.g., Llama 3, ChatML) usually fixes it."
    )
    host._native_chat_format_items = [
        ("Auto (GGUF / library default)", "auto"),
        ("GGUF Jinja (tokenizer.chat_template)", "jinja"),
        ("ChatML", "chatml"),
        ("Llama 3 Instruct", "llama-3"),
        ("Mistral / Mixtral Instruct", "mistral"),
        ("Llama 2 Chat", "llama-2"),
    ]
    host._build_prestige_menu(
        host.native_chat_format_selector,
        host._native_chat_format_items,
        host._on_native_chat_format_changed,
    )
    host.native_chat_format_reset_btn = QPushButton("Reset")
    host.native_chat_format_reset_btn.setToolTip(
        "Reset to automatic template selection for the currently loaded model."
    )
    host.native_chat_format_reset_btn.clicked.connect(
        host._on_reset_native_chat_format_clicked
    )
    chat_template_row = QWidget()
    chat_template_row_layout = QHBoxLayout(chat_template_row)
    chat_template_row_layout.setContentsMargins(0, 0, 0, 0)
    chat_template_row_layout.setSpacing(8)
    chat_template_row_layout.addWidget(host.native_chat_format_selector, stretch=1)
    chat_template_row_layout.addWidget(host.native_chat_format_reset_btn)

    chat_template_form.addRow("Chat template (internal)", chat_template_row)

    host._ai_chat_template_subsection = wrap_subsection(
        chat_template_inner, anchor="chat_template"
    )
    host.advanced_chat_template_panel.layout().addWidget(host._ai_chat_template_subsection)
    host.advanced_chat_template_panel.setVisible(get_advanced_chat_template_unlocked())
    ai_form.addRow("", host.advanced_chat_template_panel)

    # --- Auxiliary cognition ---
    add_subsection_to_form(ai_form, "Auxiliary cognition", anchor="cognition")

    _adv_tip = (
        "Advanced engine controls are not for everyday use. Only enable them if you "
        "have a very powerful machine with plenty of RAM.\n\n"
        "Unlocks optional auxiliary cognition model selection. The cognition model "
        "runs on CPU RAM in parallel with your primary chat model — larger swaps "
        "(e.g. 1.5B+) reduce headroom available for conversation."
    )
    host.advanced_engine_toggle = PrestigeToggle()
    host.advanced_engine_label = QLabel("Show advanced engine settings")
    host.advanced_engine_toggle.setToolTip(_adv_tip)
    host.advanced_engine_label.setToolTip(_adv_tip)
    host.advanced_engine_info_btn = host._make_settings_info_button(_adv_tip)
    label_cluster = QWidget()
    label_cluster_layout = QHBoxLayout(label_cluster)
    label_cluster_layout.setContentsMargins(0, 0, 0, 0)
    label_cluster_layout.setSpacing(6)
    label_cluster_layout.addWidget(host.advanced_engine_label)
    label_cluster_layout.addWidget(host.advanced_engine_info_btn)
    advanced_row = QWidget()
    advanced_row_layout = QHBoxLayout(advanced_row)
    advanced_row_layout.setContentsMargins(0, 0, 0, 0)
    advanced_row_layout.setSpacing(8)
    advanced_row_layout.addWidget(
        host.advanced_engine_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    advanced_row_layout.addWidget(label_cluster)
    advanced_row_layout.addStretch(1)
    host.advanced_engine_toggle.blockSignals(True)
    host.advanced_engine_toggle.setChecked(get_advanced_engine_unlocked())
    host.advanced_engine_toggle.blockSignals(False)
    host.advanced_engine_toggle.toggled.connect(host._on_advanced_engine_toggled)
    ai_form.addRow("", advanced_row)

    host.advanced_engine_panel = QWidget()
    adv_panel_layout = QVBoxLayout(host.advanced_engine_panel)
    adv_panel_layout.setContentsMargins(0, 8, 0, 0)
    adv_panel_layout.setSpacing(12)

    cognition_dir = get_cognition_models_dir()
    host.cognition_dir_label = QLabel(cognition_dir)
    host.cognition_dir_label.setWordWrap(True)
    host.cognition_dir_label.setToolTip(
        "Place optional cognition .gguf files here. The bundled Qwen3 1.7B default "
        "also lives in this folder."
    )

    cognition_row = QHBoxLayout()
    host.cognition_gguf_list = QListWidget()
    host.cognition_gguf_list.setMinimumHeight(90)
    host.cognition_gguf_list.setMaximumHeight(140)
    host.cognition_gguf_list.setToolTip(
        "Built-in Qwen3 1.7B default cannot be deleted. Select a custom model and "
        "click Use selected, or Reset to default."
    )
    cognition_row.addWidget(host.cognition_gguf_list, stretch=1)
    cognition_btn_col = QVBoxLayout()
    cognition_btn_col.setSpacing(8)
    host.use_cognition_gguf_btn = QPushButton("Use selected")
    apply_brand_primary(host.use_cognition_gguf_btn)
    host.use_cognition_gguf_btn.clicked.connect(host._apply_selected_cognition_gguf)
    cognition_btn_col.addWidget(
        host.use_cognition_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    host.reset_cognition_btn = QPushButton("Reset to default")
    apply_brand_primary(host.reset_cognition_btn, icon_name="fa5s.undo")
    host.reset_cognition_btn.clicked.connect(host._reset_cognition_to_default)
    cognition_btn_col.addWidget(
        host.reset_cognition_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    host.delete_cognition_gguf_btn = QPushButton("Delete")
    apply_brand_danger(host.delete_cognition_gguf_btn)
    host.delete_cognition_gguf_btn.clicked.connect(host._delete_selected_cognition_gguf)
    cognition_btn_col.addWidget(
        host.delete_cognition_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    cognition_row.addLayout(cognition_btn_col)

    host.cognition_chat_format_selector = SelectorButton(
        "Cognition chat template...", is_dark=is_dark
    )
    host.cognition_chat_format_selector.setMaximumWidth(350)
    host.cognition_chat_format_selector.setMenu(
        QMenu(host.cognition_chat_format_selector)
    )
    host.cognition_chat_format_selector.setToolTip(
        "Prompt format for the auxiliary cognition model. Auto infers from filename."
    )
    host._cognition_chat_format_items = [
        ("Auto (from filename)", "auto"),
        ("ChatML", "chatml"),
        ("Llama 3 Instruct", "llama-3"),
        ("Phi-3", "phi"),
        ("Gemma", "gemma"),
    ]
    host._build_prestige_menu(
        host.cognition_chat_format_selector,
        host._cognition_chat_format_items,
        host._on_cognition_chat_format_changed,
    )
    host._sync_cognition_chat_format_label()

    host.active_cognition_model_lbl = QLabel()
    host.active_cognition_model_lbl.setWordWrap(True)

    adv_panel_layout.addWidget(QLabel("Optional cognition models directory:"))
    adv_panel_layout.addWidget(host.cognition_dir_label)
    adv_panel_layout.addLayout(cognition_row)
    adv_panel_layout.addWidget(QLabel("Cognition chat template (advanced)"))
    adv_panel_layout.addWidget(host.cognition_chat_format_selector)
    adv_panel_layout.addWidget(host.active_cognition_model_lbl)

    host.advanced_engine_panel.setVisible(get_advanced_engine_unlocked())
    host._ai_cognition_subsection = wrap_subsection(
        host.advanced_engine_panel, anchor="cognition"
    )
    ai_form.addRow("", host._ai_cognition_subsection)

    host._wire_llm_generation_settings()
    host._refresh_cognition_gguf_list()
    host._sync_active_cognition_label()
    host._sync_native_chat_template_label()

    add_section_reset_footer(ai_form, host, "ai.models", is_dark=is_dark)

    return ai_widget
