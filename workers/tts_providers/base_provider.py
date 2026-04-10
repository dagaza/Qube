from abc import ABC, abstractmethod
from typing import Generator, Callable, Optional
import numpy as np
import logging

logger = logging.getLogger("Qube.TTS.Provider")


class BaseTTSProvider(ABC):
    """
    MASTER CONTRACT FOR ALL TTS PROVIDERS IN QUBE

    This version is STRICT:
    - prevents silent failures
    - enforces streaming correctness
    - standardizes voice behavior
    - protects against empty generators (core bug class you hit with F5)
    """

    def __init__(self):
        # ----------------------------
        # CORE STATE
        # ----------------------------
        self.is_loaded = False
        self.sample_rate = 24000

        # ----------------------------
        # CAPABILITY FLAGS
        # ----------------------------
        self.supports_voice_selection = False
        self.available_voices = ["default"]
        self.current_voice = "default"

    # =========================================================
    # REQUIRED API
    # =========================================================
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load model weights.

        MUST:
        - accept model_path
        - return True if successful
        - set self.is_loaded = True internally
        """
        pass

    @abstractmethod
    def set_voice(self, voice_name: str) -> bool:
        """
        Set or update voice / clone profile.

        MUST:
        - never silently fail
        - update self.current_voice
        """
        pass

    @abstractmethod
    def generate_audio_stream(
        self,
        text: str,
        cancel_flag_callback: Callable[[], bool],
        voice_context: Optional[object] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        CORE STREAMING INTERFACE

        HARD CONTRACT (NON-NEGOTIABLE):

        1. MUST yield at least one np.ndarray chunk
        2. MUST NOT return an empty generator silently
        3. MUST raise Exception on failure (no silent return)
        4. MUST check cancel_flag_callback frequently
        5. Each yield MUST be float32/float16 waveform [-1, 1]
        """
        pass

    # =========================================================
    # OPTIONAL BUT STANDARDIZED BEHAVIOR
    # =========================================================
    def unload_model(self):
        """
        Free memory safely when switching providers.
        """
        self.is_loaded = False
        logger.info(f"Unloaded {self.__class__.__name__}")

    # =========================================================
    # SAFETY UTILITIES (USED BY ALL PROVIDERS)
    # =========================================================
    def _validate_audio_chunk(self, wav: np.ndarray, name="chunk"):
        """
        Prevents silent corruption (NaNs, empties, wrong shape).
        """
        if wav is None:
            raise ValueError(f"{name}: audio is None")

        if not isinstance(wav, np.ndarray):
            raise TypeError(f"{name}: expected np.ndarray, got {type(wav)}")

        if wav.size == 0:
            raise ValueError(f"{name}: empty audio buffer")

        if np.isnan(wav).any():
            raise ValueError(f"{name}: NaN detected in audio")

        if np.isinf(wav).any():
            raise ValueError(f"{name}: Inf detected in audio")

        return True

    def _ensure_generator_output(self, yielded_any: bool):
        """
        Called by workers or providers to enforce:
        NO SILENT FAILURE GENERATORS.
        """
        if not yielded_any:
            raise RuntimeError(
                f"{self.__class__.__name__} produced no audio output (silent failure blocked)"
            )