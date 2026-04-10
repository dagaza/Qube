from PyQt6.QtCore import QThread, pyqtSignal
import requests
import json
import time
import re
import logging

# Import our tools
from mcp.rag_tool import rag_search
from mcp.internet_tool import search_internet
from workers.intent_router import Intent, IntentRegistry, IntentRouter, EmbeddingCache, build_centroid
from mcp.memory_tool import memory_search

logger = logging.getLogger("Qube.LLM")

class LLMWorker(QThread):
    # 🔑 Carry the session_id so the Mouth (TTS) knows the context
    sentence_ready = pyqtSignal(str, str)   
    token_streamed = pyqtSignal(str)   
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float) 
    context_retrieved = pyqtSignal(bool)
    response_finished = pyqtSignal(str, str) 
    sources_found = pyqtSignal(list)
    
    def __init__(self, embedder, store, db_manager):
        super().__init__()
        self.prompt = ""
        self.session_id = None 
        self.api_url = "http://localhost:1234/v1/chat/completions"
        self.embedder = embedder
        self.store = store
        self.db = db_manager
        
        # --- NEW: Single-Pass Embedding Cache ---
        self.embedding_cache = EmbeddingCache(self.embedder)
        
        # --- NEW: Cache DB Triggers on Boot to prevent latency ---
        try:
            self.cached_custom_triggers = [t.lower() for t in self.db.get_rag_triggers()]
        except Exception as e:
            logger.error(f"Failed to cache custom RAG triggers: {e}")
            self.cached_custom_triggers = []

        # --- NEW: Initialize the Multi-Intent Semantic Router (v3) ---
        logger.info("Building multi-intent centroids...")
        
        rag_examples = [
            "search my notes", "find my documents", "scan local files",
            "what did I write about", "retrieve my research", "check my vault",
            "look in my library", "read the document"
        ]
        web_examples = [
            "search the internet", "look up online", "who won the game yesterday", 
            "what is the current news", "google search", "live web", "current weather"
        ]
        chat_examples = [
            "what is the meaning of life", "explain quantum physics",
            "write a poem", "how does gravity work", "hello", "who are you"
        ]
        
        rag_centroid = build_centroid(self.embedder, rag_examples)
        web_centroid = build_centroid(self.embedder, web_examples)
        chat_centroid = build_centroid(self.embedder, chat_examples)
        
        registry = IntentRegistry()
        registry.register(Intent(name="RAG", centroid=rag_centroid, threshold=0.65, margin=0.03)) # Lowered margin
        registry.register(Intent(name="WEB", centroid=web_centroid, threshold=0.65, margin=0.03)) # Lowered margin
        registry.register(Intent(name="CHAT", centroid=chat_centroid, threshold=0.50, margin=0.02))
        
        self.intent_router = IntentRouter(registry)

        # UI Toggles
        self.mcp_auto_enabled = True 
        self.temperature = 0.7
        self.context_window = 4096
        self.mcp_rag_enabled = True
        self.mcp_strict_enabled = False 
        self.mcp_internet_enabled = False

    def refresh_custom_triggers(self):
        """Called by the UI when the user edits custom triggers in Settings."""
        try:
            self.cached_custom_triggers = [t.lower() for t in self.db.get_rag_triggers()]
            logger.info("In-memory custom triggers refreshed.")
        except Exception as e:
            logger.error(f"Failed to refresh custom triggers: {e}")
    
    def cancel_generation(self):
        """Triggered by main.py to sever the LLM stream."""
        logger.info("LLM Worker: Cancel requested by user interruption.")
        self._cancel_requested = True

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
        logger.info("Model reload triggered by UI. (Requires specific provider endpoint)")
        self.status_update.emit("Model Context Updated")

    def generate_response(self, text: str, session_id: str):
        """Sets the parameters and starts the thread work."""
        # Prestige Tip: Check if already running to prevent double-starts
        if self.isRunning():
            return

        self.prompt = text
        self.session_id = session_id
        self.start() # This automatically triggers the run() method below

    def clean_text_for_tts(self, text):
        text = re.sub(r'[*_]{1,3}', '', text)
        text = re.sub(r'#+\s+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text.strip()

    def _determine_execution_path(self) -> str:
        """v3 Semantic Orchestrator: Returns 'RAG', 'WEB', or 'CHAT'"""
        clean_prompt = self.prompt.lower().strip()

        # 1. Zero-Latency Cache Check (Custom Magic Words always force RAG)
        if getattr(self, 'mcp_auto_enabled', True):
            if any(trigger in clean_prompt for trigger in self.cached_custom_triggers):
                logger.info("ORCHESTRATOR: Custom trigger matched. Forcing RAG.")
                # Respect the master toggle even if they use a magic word
                return "RAG" if getattr(self, 'mcp_rag_enabled', False) else "CHAT"

        # 2. v3 Semantic Scoring Engine
        try:
            user_vec = self.embedding_cache.get_embedding(self.prompt)
            intent_name, confidence = self.intent_router.route(user_vec)
            logger.info(f"ORCHESTRATOR: Semantic Engine selected [{intent_name}] with {confidence:.2f} confidence.")
            
            # 3. The Permission Gate
            if intent_name == "RAG" and not getattr(self, 'mcp_rag_enabled', False):
                logger.debug("ORCHESTRATOR: RAG selected but Master Toggle is OFF. Falling back to CHAT.")
                return "CHAT"
                
            if intent_name == "WEB" and not getattr(self, 'mcp_internet_enabled', False):
                logger.debug("ORCHESTRATOR: WEB selected but Master Toggle is OFF. Falling back to CHAT.")
                return "CHAT"
                
            return intent_name
            
        except Exception as e:
            logger.error(f"Semantic Routing failed, falling back to CHAT: {e}")
            return "CHAT"

    def _get_available_tools(self) -> list:
        """Constructs the OpenAI-compatible tools schema based on UI toggles and NLP."""
        tools = []
        
        # 🔑 THE FIX: Use our new NLP router instead of the static boolean
        if self._should_trigger_rag():
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_local_documents",
                    "description": "Searches the user's private knowledge base/library for specific information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query to find in the documents."}
                        },
                        "required": ["query"]
                    }
                }
            })
            
        if self.mcp_internet_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_internet",
                    "description": "Searches the live internet for up-to-date information, news, or facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search engine query."}
                        },
                        "required": ["query"]
                    }
                }
            })
            
        return tools
    
    def generate(self, prompt: str) -> str:
        """
        Synchronous, non-streaming generation for background tasks.
        Used by the EnrichmentWorker to quietly extract memories.
        """
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature ensures strict JSON formatting
            "max_tokens": 1000,
            "stream": False      # 🔑 Crucial: We do NOT stream this background task
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            # Note: We use a timeout so a stalled LLM doesn't freeze the background thread
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"Background LLM API Error {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Background LLM Network Error: {e}")
            return ""
        
    def run(self):
        self._cancel_requested = False
        
        if self.session_id:
            self.db.add_message(self.session_id, "user", self.prompt)

        history = self.db.get_session_history(self.session_id) if self.session_id else []
        
        # ============================================================
        # 1. INTENT ROUTING PHASE
        # ============================================================
        self.status_update.emit("Thinking...")
        
        # 🔑 FIX: Generate vector once for the whole turn
        query_vector = self.embedding_cache.get_embedding(self.prompt)
        
        # 🔑 FIX: Use 'self.intent_router' (the correct name from your __init__)
        # Also using .route() which matches your router's API
        execution_path, confidence = self.intent_router.route(query_vector)
        logger.info(f"ORCHESTRATOR: Selected [{execution_path}] with {confidence:.2f} confidence.")
        
        # Initialize context containers
        tool_context = ""
        memory_context = "" 
        all_ui_sources = [] 
        
        # ============================================================
        # 2. TOOL EXECUTION PHASE (Pre-LLM)
        # ============================================================

        # --- A. THE MEMORY READ LAYER (Runs on CHAT and RAG) ---
        if execution_path in ["CHAT", "RAG"]:
            logger.debug("[LLM Worker] Executing zero-cost memory scan...")
            mem_result = memory_search(self.prompt, query_vector, self.store)
            memory_context = mem_result.get("memory_context", "")
            all_ui_sources.extend(mem_result.get("memory_sources", []))

        # --- B. RAG LAYER ---
        if execution_path == "RAG":
            self.status_update.emit("Searching Local Library...")
            result_data = rag_search(self.prompt, query_vector, self.store)
            tool_context = result_data.get("llm_context", "")
            all_ui_sources.extend(result_data.get("sources", []))
            
            if not tool_context:
                tool_context = "No relevant local documents found."
                
        # --- C. WEB LAYER ---
        elif execution_path == "WEB":
            self.status_update.emit("Searching the Internet...")
            tool_context = search_internet(self.prompt)
            if not tool_context:
                tool_context = "Web search returned no results."

        # Emit ALL sources (Memory pills + RAG pills) to the UI at once
        if all_ui_sources:
            self.sources_found.emit(all_ui_sources)

        # ============================================================
        # 3. PROMPT COMPILATION PHASE (Unified Context Injector)
        # ============================================================
        base_system_prompt = (
            "You are Qube, an advanced, highly capable AI research assistant. "
            "You are helpful, concise, and have broad general knowledge."
        )
        
        if execution_path == "RAG":
            base_system_prompt += (
                " You have been provided with documents from the user's local library. "
                "You MUST include inline citations at the end of relevant sentences using brackets, like [1] or [2]. "
            )
            if getattr(self, 'mcp_strict_enabled', False):
                base_system_prompt += (
                    " STRICT ISOLATION MODE: You must NEVER use your general knowledge. "
                    "If the documents do not contain the answer, state: 'I cannot answer that based on the provided documents.'"
                )
        elif execution_path == "WEB":
             base_system_prompt += " You have been provided with live internet search results. Synthesize them to answer the user."

        messages = [{"role": "system", "content": base_system_prompt}] + history
        
        if len(messages) > 0 and messages[-1]["role"] == "user":
            original_prompt = messages[-1]["content"]
            assembled_parts = []

            if memory_context:
                assembled_parts.append(f"RELEVANT PAST MEMORIES ABOUT THE USER:\n{memory_context}")

            if tool_context:
                assembled_parts.append(f"SYSTEM TOOL OUTPUT:\n{tool_context}")

            assembled_parts.append(f"USER QUERY:\n{original_prompt}")
            
            if tool_context:
                assembled_parts.append("Please answer my query based on the tool results provided above. Incorporate my past memories if they are relevant.")
            elif memory_context:
                assembled_parts.append("Please answer my query using your general knowledge, while keeping my past memories in mind.")

            messages[-1]["content"] = "\n\n---\n\n".join(assembled_parts)

        # ============================================================
        # 4. LLM SYNTHESIS PHASE
        # ============================================================
        self.status_update.emit("Synthesizing...")
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.context_window,
            "stream": True
        }
        
        headers = {"Content-Type": "application/json"}
        current_sentence = ""
        final_assistant_response = ""
        first_token_received = False
        start_time = time.time()

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, stream=True)
            if response.status_code != 200:
                self.status_update.emit("LLM API Error")
                return
            
            for line in response.iter_lines():
                if getattr(self, '_cancel_requested', False): break 
                if not line: continue
                decoded = line.decode('utf-8')
                
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]": break
                    
                    try:
                        packet = json.loads(data_str)
                        new_text = packet['choices'][0].get('delta', {}).get('content', "")
                        
                        if new_text:
                            if not first_token_received:
                                self.ttft_latency.emit((time.time() - start_time) * 1000)
                                first_token_received = True

                            current_sentence += new_text
                            final_assistant_response += new_text
                            self.token_streamed.emit(new_text) 
                            
                            punctuation_marks = ['.', '!', '?']
                            if any(p in new_text for p in punctuation_marks):
                                clean = self.clean_text_for_tts(current_sentence.strip())
                                # 🔑 UPDATE: Send the session_id to the Mouth (TTS)
                                if clean: self.sentence_ready.emit(clean, self.session_id)
                                current_sentence = ""
                    except json.JSONDecodeError:
                        pass

            # Final Flush
            if current_sentence.strip() and not getattr(self, '_cancel_requested', False):
                clean = self.clean_text_for_tts(current_sentence.strip())
                # 🔑 UPDATE: Send the session_id to the Mouth (TTS)
                if clean: self.sentence_ready.emit(clean, self.session_id)

        except Exception as e:
            logger.error(f"LLM Network Error: {e}")

        # 5. PERSISTENCE
        if self.session_id and final_assistant_response.strip():
            self.db.add_message(self.session_id, "assistant", final_assistant_response.strip())
            self.response_finished.emit(self.session_id, final_assistant_response.strip())
            
        self.status_update.emit("Idle")