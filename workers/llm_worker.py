from PyQt6.QtCore import QThread, pyqtSignal
import requests
import json
import time
import re
import logging

# FIX 1: Give this worker its proper identity!
logger = logging.getLogger("Qube.LLM")

class LLMWorker(QThread):
    sentence_ready = pyqtSignal(str)   
    token_streamed = pyqtSignal(str)   
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float) 
    context_retrieved = pyqtSignal(bool) 

    def __init__(self, embedder, store):
        super().__init__()
        self.prompt = ""
        self.api_url = "http://localhost:1234/v1/chat/completions"
        self.embedder = embedder
        self.store = store
        
        # RAG Routing State
        self.dashboard_rag_enabled = False
        self.auto_fallback_enabled = False
        self.nlp_trigger_enabled = False
        self.nlp_keywords = ["check my notes", "in my documents", "what did the file say"]

    def set_provider(self, port):
        self.api_url = f"http://localhost:{port}/v1/chat/completions"
        self.status_update.emit(f"Switched LLM Provider (Port: {port})")

    def generate_response(self, text):
        self.prompt = text
        self.start()

    def set_dashboard_rag(self, enabled: bool):
        self.dashboard_rag_enabled = enabled
        logger.debug(f"Dashboard Force-RAG toggled to: {enabled}")

    def set_auto_fallback(self, enabled: bool):
        self.auto_fallback_enabled = enabled
        logger.debug(f"Auto-Fallback RAG toggled to: {enabled}")

    def set_nlp_trigger(self, enabled: bool):
        self.nlp_trigger_enabled = enabled
        logger.debug(f"NLP Trigger RAG toggled to: {enabled}")

    def set_nlp_keywords(self, keywords_str: str):
        self.nlp_keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]

    def clean_text_for_tts(self, text):
        text = re.sub(r'[*_]{1,3}', '', text)
        text = re.sub(r'#+\s+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text.strip()

    def run(self):
        self.status_update.emit("Thinking...")
        logger.info(f"Received User Prompt: '{self.prompt}'")
        
        # --- THE ROUTING CASCADE ---
        attempt_rag_search = False
        
        if self.dashboard_rag_enabled:
            logger.info("RAG triggered by Dashboard Override.")
            attempt_rag_search = True
        elif self.nlp_trigger_enabled:
            prompt_lower = self.prompt.lower()
            if any(kw in prompt_lower for kw in self.nlp_keywords):
                logger.info("RAG triggered by NLP Keyword match.")
                attempt_rag_search = True
        elif self.auto_fallback_enabled:
            logger.info("RAG triggered by Auto-Fallback (Always Search).")
            attempt_rag_search = True

        context = ""
        
        # FIX 2: Native RAG Search (No external black-box dependency)
        if attempt_rag_search:
            logger.info("Executing database vector search...")
            try:
                # Turn prompt into vector
                query_vector = self.embedder.embed([self.prompt])[0]
                
                # Search database for top 5 chunks
                results = self.store.search(query_vector, top_k=5)
                
                if results:
                    # Format nicely for the LLM
                    context_chunks = []
                    for r in results:
                        context_chunks.append(f"--- Source: {r['source']} ---\n{r['text']}")
                    
                    context = "\n\n".join(context_chunks)
                    logger.info(f"RAG Success: Extracted {len(results)} chunks from database.")
                else:
                    logger.warning("Database searched, but 0 chunks were returned.")
            except Exception as e:
                logger.error(f"RAG Database Search Crashed: {e}", exc_info=True)
        
        # FIX 3: Construct the final payload and log the evidence
        if context:
            system_prompt = (
                "You are HAL. The following context has been retrieved from the user's "
                "personal document library. Answer using this context. "
                "If the answer is not present in the context, say you do not know.\n\n"
                f"CONTEXT:\n{context}"
            )
            self.context_retrieved.emit(True)
            logger.debug(f"System Prompt injected with {len(context)} characters of context.")
        else:
            system_prompt = "You are a helpful, concise AI assistant."
            self.context_retrieved.emit(False)
            logger.debug("System Prompt: Standard AI (No Context Injected).")

        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.prompt}
            ],
            "temperature": 0.7,
            "stream": True
        }

        punctuation_marks = ['.', '!', '?']
        current_sentence = ""
        first_token_received = False 
        start_time = time.time()     

        try:
            logger.debug(f"Transmitting POST request to {self.api_url}")
            response = requests.post(self.api_url, headers=headers, json=payload, stream=True)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            packet = json.loads(data_str)
                            delta = packet['choices'][0].get('delta', {})
                            new_text = delta.get('content', "")
                            
                            if new_text:
                                if not first_token_received:
                                    latency = (time.time() - start_time) * 1000
                                    self.ttft_latency.emit(latency)
                                    logger.info(f"Time to First Token: {int(latency)}ms")
                                    first_token_received = True

                                current_sentence += new_text
                                self.token_streamed.emit(new_text) 
                                
                                if any(p in new_text for p in punctuation_marks):
                                    clean_sentence = self.clean_text_for_tts(current_sentence.strip())
                                    if clean_sentence:
                                        self.sentence_ready.emit(clean_sentence)
                                    current_sentence = ""
                                    
                        except json.JSONDecodeError:
                            pass
            
            if current_sentence.strip():
                clean_sentence = self.clean_text_for_tts(current_sentence.strip())
                if clean_sentence:
                    self.sentence_ready.emit(clean_sentence)
            
            logger.info("LLM Generation Complete.")
                
        except Exception as e:
            self.status_update.emit(f"LLM Error: {e}")
            logger.error(f"HTTP Request to LLM Provider failed: {e}")
            
        self.status_update.emit("Idle")