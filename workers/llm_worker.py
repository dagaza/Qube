from PyQt6.QtCore import QThread, pyqtSignal
import requests
import json
import time
import re
import logging

# Import our tools
from mcp.rag_tool import rag_search
from mcp.internet_tool import search_internet

logger = logging.getLogger("Qube.LLM")

class LLMWorker(QThread):
    sentence_ready = pyqtSignal(str)   
    token_streamed = pyqtSignal(str)   
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float) 
    context_retrieved = pyqtSignal(bool)
    response_finished = pyqtSignal(str, str) 
    sources_found = pyqtSignal(list) # Will emit the list of dicts
    
    def __init__(self, embedder, store, db_manager):
        super().__init__()
        self.prompt = ""
        self.session_id = None 
        self.api_url = "http://localhost:1234/v1/chat/completions"
        self.embedder = embedder
        self.store = store
        self.db = db_manager
        
        # --- NEW: Dynamic Generation Parameters ---
        self.temperature = 0.7
        self.context_window = 4096
        
        # --- NEW: MCP Tool States ---
        self.mcp_rag_enabled = True
        self.mcp_internet_enabled = False

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

    def _get_available_tools(self) -> list:
        """Constructs the OpenAI-compatible tools schema based on UI toggles."""
        tools = []
        if self.mcp_rag_enabled:
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

    def run(self):
        self.status_update.emit("Thinking...")
        
        if self.session_id:
            self.db.add_message(self.session_id, "user", self.prompt)

        # 1. THE PRESTIGE FIX: Build Initial History with Strict Citation Rules
        history = self.db.get_session_history(self.session_id) if self.session_id else []
        system_prompt = (
            "You are Qube, an advanced AI research assistant. "
            "When you use the 'search_local_documents' tool, you will receive information labeled as 'SOURCE 1', 'SOURCE 2', etc. "
            "If you use information from these sources to answer the user, you MUST include inline citations at the end of the relevant sentences using brackets, like this: [1] or [2]. "
            "Never invent or hallucinate sources. If you do not know the answer based on the sources, say so."
        )
        messages = [{"role": "system", "content": system_prompt}] + history
        
        # We use a loop because if the LLM calls a tool, we need to feed the result back to it
        max_tool_iterations = 3 
        current_iteration = 0
        final_assistant_response = ""

        while current_iteration < max_tool_iterations:
            current_iteration += 1
            
            payload = {
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.context_window,
                "stream": True
            }
            
            # Inject tools if enabled
            active_tools = self._get_available_tools()
            if active_tools:
                payload["tools"] = active_tools

            headers = {"Content-Type": "application/json"}
            current_sentence = ""
            first_token_received = False
            start_time = time.time()
            
            # Buffers for intercepting streaming tool calls
            is_tool_call = False
            tool_name = ""
            tool_args_str = ""
            tool_call_id = "call_123" # Fallback ID

            try: # <--- THE OUTER TRY BLOCK
                response = requests.post(self.api_url, headers=headers, json=payload, stream=True)
                for line in response.iter_lines():
                    if not line: continue
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str.strip() == "[DONE]": break
                        
                        try:
                            packet = json.loads(data_str)
                            delta = packet['choices'][0].get('delta', {})
                            
                            # INTERCEPT TOOL CALLS
                            if 'tool_calls' in delta:
                                is_tool_call = True
                                tc = delta['tool_calls'][0]
                                if 'id' in tc: tool_call_id = tc['id']
                                if 'function' in tc:
                                    if 'name' in tc['function']: tool_name = tc['function']['name']
                                    if 'arguments' in tc['function']: tool_args_str += tc['function']['arguments']
                                continue

                            # NORMAL TEXT STREAMING
                            new_text = delta.get('content', "")
                            if new_text and not is_tool_call:
                                if not first_token_received:
                                    if hasattr(self, 'ttft_latency'):
                                        self.ttft_latency.emit((time.time() - start_time) * 1000)
                                    first_token_received = True

                                current_sentence += new_text
                                final_assistant_response += new_text
                                self.token_streamed.emit(new_text) 
                                
                                punctuation_marks = ['.', '!', '?']
                                if any(p in new_text for p in punctuation_marks):
                                    clean = self.clean_text_for_tts(current_sentence.strip())
                                    if clean: self.sentence_ready.emit(clean)
                                    current_sentence = ""
                                    
                        except json.JSONDecodeError:
                            pass

                # Flush remaining text
                if current_sentence.strip() and not is_tool_call:
                    clean = self.clean_text_for_tts(current_sentence.strip())
                    if clean: self.sentence_ready.emit(clean)

                # --- TOOL EXECUTION LOGIC ---
                if is_tool_call:
                    self.status_update.emit(f"Using tool: {tool_name}...")
                    try:
                        args = json.loads(tool_args_str)
                        query = args.get("query", "")
                        
                        # 1. Append the AI's intent to call the tool to history
                        messages.append({
                            "role": "assistant", 
                            "content": None, 
                            "tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": tool_name, "arguments": tool_args_str}}]
                        })

                        # 2. Execute the requested tool
                        llm_text_payload = "" 

                        if tool_name == "search_local_documents":
                            # It returns our new dictionary
                            result_data = rag_search(query, self.embedder, self.store)
                            
                            # A. Extract the text for the LLM
                            llm_text_payload = result_data["llm_context"]
                            
                            # B. Emit the metadata to the UI for the source chips
                            self.sources_found.emit(result_data["sources"])
                            
                            # C. Tell the top bar if we found anything
                            self.context_retrieved.emit(bool(llm_text_payload))
                            
                        elif tool_name == "search_internet":
                            # Assuming this still returns a standard string
                            llm_text_payload = search_internet(query)

                        # Fallback if the database/internet is empty
                        if not llm_text_payload:
                            llm_text_payload = "No results found."

                        # 3. Append the clean text payload to the LLM's history
                        messages.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": llm_text_payload
                        })
                        
                        # 🔑 THE NUDGE FIX: Force the local model to answer
                        messages.append({
                            "role": "system",
                            "content": "Tool executed successfully. Please provide the final answer to the user based ONLY on the tool output above. Do not call the tool again."
                        })
                        
                        logger.info(f"Tool '{tool_name}' executed. Looping back to LLM for final answer.")
                        continue # Loop back up and POST to the LLM again!
                        
                    except Exception as e:
                        logger.error(f"Failed to execute tool {tool_name}: {e}")
                        break # Break loop on failure
                else:
                    # Normal generation completed, break the iteration loop
                    break 
                    
            except Exception as e: # <--- THE MISSING EXCEPT BLOCK RESTORED
                logger.error(f"LLM Network Error: {e}")
                break
                
        # Save final text to SQLite
        if self.session_id and final_assistant_response.strip():
            # 1. Save to DB first so history is ready for the TitleWorker
            self.db.add_message(self.session_id, "assistant", final_assistant_response.strip())
            
            # 2. THE PRESTIGE EMIT: Tell the app we are done talking
            self.response_finished.emit(self.session_id, final_assistant_response.strip())
            
        self.status_update.emit("Idle")