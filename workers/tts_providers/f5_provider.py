import os
import torch
import numpy as np
import logging

from .base_provider import BaseTTSProvider

logger = logging.getLogger("Qube.TTS.F5")


class F5Provider(BaseTTSProvider):
    def __init__(self):
        super().__init__()

        self.supports_voice_selection = False
        self.available_voices = []

        self.model = None
        self.sample_rate = 24000

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.reference_audio_path = None
        self.reference_text = ""

        logger.info(f"F5 initialized on {self.device}")

    def load_model(self, model_path: str) -> bool:
        try:
            from f5_tts.api import F5TTS

            self.model = F5TTS(
                ckpt_file=model_path,
                device=self.device
            )

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"F5 load failed: {e}", exc_info=True)
            return False

    def set_voice(self, voice_path: str):
        if not os.path.exists(voice_path):
            logger.error("Voice file missing")
            return False

        self.reference_audio_path = voice_path
        self.reference_text = "voice sample"
        return True

    def generate_audio_stream(self, text: str, cancel_flag_callback):
        try:
            if not self.model:
                raise RuntimeError("F5 not loaded")

            if not self.reference_audio_path:
                logger.error("F5 missing reference audio")
                return

            # SAFE INFER CALL (NO ASSUMPTION ABOUT API)
            logger.info("F5 inference starting")

            try:
                result = self.model.infer(
                    self.reference_audio_path,
                    self.reference_text,
                    text
                )
            except TypeError:
                # fallback signature attempt
                result = self.model.infer(text)

            if cancel_flag_callback():
                return

            if isinstance(result, tuple):
                wav = result[0]
            else:
                wav = result

            if wav is None:
                return

            if torch.is_tensor(wav):
                wav = wav.detach().cpu().numpy()

            wav = np.clip(wav, -1.0, 1.0)

            pcm = (wav * 32767).astype(np.int16).tobytes()

            for i in range(0, len(pcm), 4096):
                if cancel_flag_callback():
                    break
                yield pcm[i:i+4096]

        except Exception as e:
            logger.error(f"F5 error: {e}", exc_info=True)

    def unload_model(self):
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()