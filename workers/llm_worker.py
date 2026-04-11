from PyQt6.QtCore import QThread, pyqtSignal
import requests
import json
import time
import re
import logging

from mcp.rag_tool import rag_search
from mcp.internet_tool import search_internet
from mcp.memory_tool import memory_search
from workers.intent_router import EmbeddingCache

from mcp.cognitive_router import CognitiveRouterV4
from mcp.router_telemetry import RouterTelemetryBrain
from mcp.router_self_tuner import AdaptiveRouterSelfTunerV2

logger = logging.getLogger("Qube.LLM")


class LLMWorker(QThread):
    sentence_ready = pyqtSignal(str, str)
    token_streamed = pyqtSignal(str)
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float)
    context_retrieved = pyqtSignal(bool)
    response_finished = pyqtSignal(str, str)
    sources_found = pyqtSignal(list)
    router_telemetry_updated = pyqtSignal(dict, dict)  # summary, tuner_state

    MAX_TOTAL_RETRIEVAL_CHARS = 4500
    MEMORY_BUDGET = 1500
    RAG_BUDGET = 3000

    def __init__(self, embedder, store, db_manager):
        super().__init__()

        self.prompt = ""
        self.session_id = None
        self.api_url = "http://localhost:1234/v1/chat/completions"

        self.embedder = embedder
        self.store = store
        self.db = db_manager

        self.embedding_cache = EmbeddingCache(self.embedder)

        # triggers
        try:
            self.cached_custom_triggers = [
                t.lower() for t in self.db.get_rag_triggers()
            ]
        except Exception:
            self.cached_custom_triggers = []

        # ================================
        # BRAIN STACK
        # ================================
        self.cognitive_router = CognitiveRouterV4()
        self.telemetry = RouterTelemetryBrain()
        self.router_tuner = AdaptiveRouterSelfTunerV2()

        self.USE_COGNITIVE_ROUTER = True
        self.USE_ADAPTIVE_ROUTER = True
        self.USE_TELEMETRY = True
        self.USE_COGNITIVE_ROUTER_INTERNET = True # For hybrid internet mode

        # toggles
        self.mcp_auto_enabled = True
        self.temperature = 0.7
        self.context_window = 4096
        self.mcp_rag_enabled = True
        self.mcp_strict_enabled = False
        self.mcp_internet_enabled = False

    # ============================================================
    # RETRIEVAL BUDGET ENFORCER
    # ============================================================
    def _enforce_retrieval_budget(self, memory_context: str, rag_context: str):

        def trim(t, limit):
            return t[:limit] if t else ""

        memory_context = trim(memory_context, self.MEMORY_BUDGET)

        remaining = self.MAX_TOTAL_RETRIEVAL_CHARS - len(memory_context)
        remaining = max(0, remaining)

        rag_context = trim(rag_context, min(self.RAG_BUDGET, remaining))

        return memory_context, rag_context

    # ============================================================
    def clean_text_for_tts(self, text):
        import re
        text = re.sub(r'[*_]{1,3}', '', text)
        text = re.sub(r'#+\s+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Strip HTML and Citations (for RAG/Web)
        text = re.sub(r'<[^>]+>', '', text) 
        text = re.sub(r'\[(\d+|W)\]', '', text) 
        
        cleaned = text.strip()
        
        # 🔑 THE ULTIMATE FAILSAFE: 
        # If the string contains no letters or numbers (e.g., it's just a ".", "!", or empty), kill it.
        if not re.search(r'[a-zA-Z0-9]', cleaned):
            return ""
            
        return cleaned

    # ============================================================
    def generate_response(self, text: str, session_id: str):
        """Sets the parameters and starts the thread work."""
        if self.isRunning():
            return

        self.prompt = text
        self.session_id = session_id
        self.start() # This automatically triggers the run() method

    # ============================================================
    def generate(self, prompt: str) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000,
            "stream": False
        }

        try:
            r = requests.post(self.api_url, json=payload, timeout=120)
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            return ""

    # ============================================================
    def run(self):
        self._cancel_requested = False

        if self.session_id:
            self.db.add_message(self.session_id, "user", self.prompt)

        history = self.db.get_session_history(self.session_id) if self.session_id else []
        clean_prompt = self.prompt.lower().strip()

        # ============================================================
        # 1. ROUTING PHASE
        # ============================================================
        self.status_update.emit("Thinking...")

        intent_vector = None

        if self.USE_COGNITIVE_ROUTER:
            intent_vector = self.embedding_cache.get_embedding(self.prompt)
            decision = self.cognitive_router.route(
                self.prompt,
                intent_vector=intent_vector,
                weights=self.router_tuner.get_weights() if self.USE_ADAPTIVE_ROUTER else None
            )
        else:
            decision = {"route": "none", "strategy": "fallback"}

        execution_route = decision["route"].upper()

        # custom triggers override
        if self.mcp_auto_enabled:
            if any(t in clean_prompt for t in self.cached_custom_triggers):
                execution_route = "RAG"
                decision["rag_query"] = self.prompt

        # ------------------------------------------------------------
        # INTERNET TRIGGER (manual + cognitive)
        # ------------------------------------------------------------
        # Manual trigger: user text contains known web commands
        web_triggers = ["search the internet", "who won", "current news", "weather"]
        manual_web = any(t in clean_prompt for t in web_triggers) and self.mcp_internet_enabled

        # Automatic trigger: cognitive router decides internet is needed
        auto_web = getattr(self, "USE_COGNITIVE_ROUTER_INTERNET", False) and decision.get("internet_enabled", False)

        # Final execution decision for WEB
        if manual_web or auto_web:
            execution_route = "WEB"

        # ============================================================
        # ROUTING START TIME (telemetry)
        # ============================================================
        route_start = time.time()

        logger.info(f"[Router] route={execution_route}")

        # ============================================================
        # 2. TOOL EXECUTION
        # ============================================================
        memory_context = ""
        tool_context = ""
        all_ui_sources = []

        # 🔑 THE FIX: Initialize these dictionaries so telemetry doesn't crash
        mem_result = {} 
        rag_result = {}
        web_context = "" # Also initialize this to be safe

        query_vector = None

        if execution_route in ["MEMORY", "RAG", "HYBRID"]:
            query_vector = self.embedding_cache.get_embedding(self.prompt)

        # ---- MEMORY ----
        if execution_route in ["MEMORY", "HYBRID"]:
            mem_result = memory_search(
                decision.get("memory_query") or self.prompt,
                query_vector,
                self.store
            )
            memory_context = mem_result.get("memory_context", "")
            all_ui_sources.extend(mem_result.get("memory_sources", []))

        # ---- RAG ----
        if execution_route in ["RAG", "HYBRID"] and self.mcp_rag_enabled:
            rag_result = rag_search(
                decision.get("rag_query") or self.prompt,
                query_vector,
                self.store
            )
            # 🔑 Use += to ensure we don't accidentally wipe out other tool data
            tool_context += rag_result.get("llm_context", "")
            all_ui_sources.extend(rag_result.get("sources", []))

        # ---- WEB + HYBRID ----
        if execution_route in ["WEB", "INTERNET", "HYBRID"] and self.mcp_internet_enabled:
            from mcp.internet_tool import search_internet
            self.status_update.emit("🌐 Searching the Web...")
            
            web_results = search_internet(self.prompt)
            
            if web_results:
                if isinstance(web_results, list):
                    web_results = "\n\n".join([str(item) for item in web_results])
                else:
                    web_results = str(web_results)

                web_context = web_results[:self.RAG_BUDGET]
                
                all_ui_sources.append({
                    "id": "W", 
                    "filename": "Live Web Search", 
                    "content": web_context, 
                    "type": "web"
                })
                
                # 🔑 Append Web results to tool_context
                if tool_context:
                    tool_context = f"{tool_context}\n\n[W] WEB SEARCH RESULTS:\n{web_context}"
                else:
                    tool_context = f"[W] WEB SEARCH RESULTS:\n{web_context}"
                    
                logger.info(f"[LLM Worker] Web search integrated ({len(web_context)} chars)")

        # 🔑 THE CRITICAL FIX: Move this OUTSIDE of the tool blocks 
        # so it fires for RAG and Memory too!
        if all_ui_sources:
            self.sources_found.emit(all_ui_sources)
        
        # ============================================================
        # TELEMETRY + SELF TUNING
        # ============================================================
        latency_ms = (time.time() - route_start) * 1000

        if self.USE_TELEMETRY:
            self.telemetry.log({
                "route": execution_route,
                "memory_hits": len(mem_result.get("memory_sources", [])),
                "rag_hits": len(rag_result.get("sources", [])),
                "latency_ms": latency_ms,
                "memory_chars": len(memory_context),
                "rag_chars": len(tool_context),
            })

            self.router_tuner.observe({
                "route": execution_route,
                "memory_hits": len(mem_result.get("memory_sources", [])),
                "rag_hits": len(rag_result.get("sources", [])),
                "latency_ms": latency_ms,
            })
            
            try:
                summary = getattr(self.telemetry, 'get_summary', lambda: {})()
                tuner_state = self.router_tuner.get_weights() 
                self.router_telemetry_updated.emit(summary, tuner_state)
            except Exception as e:
                logger.error(f"Failed to emit router telemetry: {e}")

        # 🔑 NEW: Feed the Cognitive V4 Router its learning data!
        if self.USE_COGNITIVE_ROUTER and hasattr(self, 'cognitive_router'):
            # V4 expects latency in seconds, not milliseconds
            latency_seconds = latency_ms / 1000.0 
            # Did we actually use RAG this turn?
            rag_was_used = len(rag_result.get("sources", [])) > 0
            
            self.cognitive_router.record_latency(latency_seconds)
            self.cognitive_router.record_rag_used(rag_was_used)
            logger.debug(f"[Router Feedback] Logged latency: {latency_seconds:.2f}s | RAG used: {rag_was_used}")

        # ============================================================
        # 2.5 RETRIEVAL BUDGET ENFORCEMENT
        # ============================================================
        memory_context, tool_context = self._enforce_retrieval_budget(
            memory_context,
            tool_context
        )

        # ============================================================
        # 3. PROMPT BUILD
        # ============================================================
        system_prompt = (
            "You are Qube, a highly capable offline AI assistant. "
            "Be concise and accurate."
        )

        if execution_route in ["RAG", "HYBRID"]:
            system_prompt += " You MUST cite your sources inline using brackets and the ID, like [1] or [2]."
            
        # 🔑 THE FIX: Make the LLM hide its scratchpad and just talk normally!
        elif execution_route in ["WEB", "INTERNET"]:
            system_prompt = (
                "You are Qube. You have just been provided with real-time, live web search results. "
                "You MUST use the TOOLS context provided below to answer the user's query. "
                "Do not state that you are offline or cannot browse the internet. "
                "CRITICAL: Respond directly to the user in a natural, conversational tone. "
                "Do NOT output your internal reasoning, 'Step 1' thoughts, or search metadata. "
                "Output ONLY your final answer."
                "You MUST cite the web sources inline using brackets like [W]."
            )

        messages = [{"role": "system", "content": system_prompt}] + history

        # 🔑 THE MISSING LINK: Inject the gathered data into the LLM's prompt!
        if tool_context and messages and messages[-1]["role"] == "user":
            original_query = messages[-1]["content"]
            
            # Combine the gathered facts with the user's actual question
            messages[-1]["content"] = (
                f"=== SYSTEM RETRIEVED CONTEXT ===\n"
                f"Use the following numbered sources to answer the query. Cite them exactly as formatted.\n\n"
                f"{tool_context}\n"
                f"================================\n\n"
                f"USER QUERY:\n{original_query}"
            )
            
            logger.debug("Successfully injected tool context into the final prompt.")

        # ============================================================
        # 4. LLM STREAMING
        # ============================================================
        self.status_update.emit("Synthesizing...")

        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.context_window,
            "stream": True
        }

        current_sentence = ""
        final_text = ""
        start = time.time()
        first_token = False
        
        # 🔑 Initialize our safety flag
        self._successfully_finished = False 

        # 🔑 THE MASTER WRAPPER: try/except/finally
        try:
            # The safe request with a 10-second timeout
            r = requests.post(self.api_url, json=payload, stream=True, timeout=10)
            r.raise_for_status() # Ensure we didn't get a 404 or 500 error

            for line in r.iter_lines():
                if getattr(self, "_cancel_requested", False):
                    break

                if not line:
                    continue

                data = line.decode("utf-8")

                if data.startswith("data: "):
                    chunk = data[6:]
                    if chunk.strip() == "[DONE]":
                        break

                    try:
                        packet = json.loads(chunk)
                        delta = packet["choices"][0].get("delta", {}).get("content", "")

                        if delta:
                            if not first_token:
                                self.ttft_latency.emit((time.time() - start) * 1000)
                                first_token = True

                            current_sentence += delta
                            final_text += delta
                            self.token_streamed.emit(delta)

                            if any(p in delta for p in ".!?"):
                                clean = self.clean_text_for_tts(current_sentence)
                                if clean:
                                    self.sentence_ready.emit(clean, self.session_id)
                                current_sentence = ""

                    except json.JSONDecodeError:
                        continue

            if current_sentence.strip():
                clean = self.clean_text_for_tts(current_sentence)
                if clean: 
                    self.sentence_ready.emit(clean, self.session_id)

            # ============================================================
            # 5. MEMORY SAVE
            # ============================================================
            if self.session_id and final_text.strip():
                self.db.add_message(self.session_id, "assistant", final_text)
                
            # If we reached this line, the LLM fully finished without crashing
            self._successfully_finished = True

        except requests.exceptions.Timeout:
            logger.error("LLM Connection Error: Request timed out.")
            final_text = "Sorry, my brain disconnected (Timeout)."
            self.token_streamed.emit("\n\n*(Connection Timeout)*")
            
        except Exception as e:
            logger.error(f"LLM Connection Error: {e}")
            final_text = "Sorry, my brain encountered an error."
            self.token_streamed.emit("\n\n*(Connection Error)*")

        finally:
            # 1. Unlock the Chatbox
            self.response_finished.emit(self.session_id, final_text)
            
            # 2. 🔑 THE CONDITIONAL IDLE:
            # If we crashed, TTS will never run, so we MUST force "Idle" to clear the Top Bar.
            # If we succeeded, we stay quiet and let the audio engine handle the Top Bar!
            if not getattr(self, '_successfully_finished', False):
                self.status_update.emit("Idle")

        # 🔑 THE FIX: Delete `self.status_update.emit("Idle")` from here!  It was moved to the audio_worker.py.
        # Why we added it back in later on, but in a different form: We need the LLM to stay completely quiet on a successful run (letting the Audio worker do its job), but we need the LLM to yell "Idle!" if it crashes.
        # Let the audio engine control the final unlock so the UI doesn't flicker.

    # --- SETTERS FOR THE UI BLUEPRINT ---
    def set_provider(self, port: int):
        self.api_url = f"http://localhost:{port}/v1/chat/completions"
        self.status_update.emit(f"Switched LLM Provider (Port: {port})")
        logger.info(f"LLM Provider API URL updated to: {self.api_url}")

    def set_temperature(self, val: float):
        self.temperature = val
        logger.debug(f"Temperature updated to {val}")

    def set_context_window(self, val: int):
        self.context_window = val
        logger.debug(f"Context Window updated to {val}")

    def set_mcp_rag(self, enabled: bool):
        self.mcp_rag_enabled = enabled

    def set_mcp_strict(self, enabled: bool):
        self.mcp_strict_enabled = enabled
        logger.debug(f"Strict Isolation Mode set to: {enabled}")

    def set_mcp_auto(self, enabled: bool):
        self.mcp_auto_enabled = enabled
        logger.debug(f"NLP Auto-Activator set to: {enabled}")
        
    def set_mcp_internet(self, enabled: bool):
        self.mcp_internet_enabled = enabled

    def reload_model(self):
        """Placeholder for LM Studio / Ollama model reload endpoints."""
        logger.info("Model reload triggered by UI.")
        self.status_update.emit("Model Context Updated")