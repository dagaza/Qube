"""Background wakeword model downloads.

This keeps Settings UI responsive while fetching model assets from GitHub.
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from core.paths import models_root

logger = logging.getLogger("Qube.WakewordDownloadWorker")

COMMUNITY_REPO_ZIP_MAIN = (
    "https://github.com/fwartner/home-assistant-wakewords-collection/archive/refs/heads/main.zip"
)

OPENWAKEWORD_WAKEWORDS_TO_DOWNLOAD: list[str] = [
    # Exclude these two from the openWakeWord built-in set.
    "alexa",
    "hey_mycroft",
    "hey_jarvis",
    "hey_rhasspy",
]

# Exclude these from the community pack.
COMMUNITY_WAKEWORDS_TO_EXCLUDE: set[str] = {
    "hey_dick_head",
    "oi_fuckwhit",
    "yo_bitch",
}


class WakewordModelsDownloadWorker(QThread):
    status_message = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, kind: str):
        super().__init__()
        self._kind = (kind or "").strip().lower()

    def run(self) -> None:
        try:
            if self._kind == "openwakeword":
                self._download_openwakeword()
            elif self._kind == "community":
                self._download_community()
            else:
                raise ValueError(f"Unknown download kind: {self._kind!r}")

            self.finished_ok.emit()
        except Exception as exc:  # pragma: no cover - UI handles messaging
            logger.exception("Wakeword download failed: %s", exc)
            self.failed.emit(str(exc))

    def _download_openwakeword(self) -> None:
        self.status_message.emit("Preparing OpenWakeWord model install…")

        import openwakeword
        import openwakeword.utils

        # openWakeWord downloads into its own package dir:
        #   <site-packages>/openwakeword/resources/models/
        target_dir = (
            Path(openwakeword.utils.__file__).resolve().parent / "resources" / "models"
        )

        def _clean_stem(path_or_name: str) -> str:
            stem = Path(path_or_name).stem
            return stem.split("_v")[0].strip().lower()

        # If the wakewords we care about are already present, don't require
        # write access (important for read-only packaging scenarios).
        wanted_present = True

        # Verify both tflite + onnx variants so whichever inference path
        # openWakeWord chooses at runtime will work.
        tflite_paths = openwakeword.get_pretrained_model_paths("tflite")
        onnx_paths = openwakeword.get_pretrained_model_paths("onnx")
        id_to_tflite = {_clean_stem(p): p for p in tflite_paths}
        id_to_onnx = {_clean_stem(p): p for p in onnx_paths}

        # Also verify feature models used by AudioFeatures so the preprocessor works.
        # openWakeWord's download helper always downloads these.
        feature_models = []
        for feature in getattr(openwakeword, "FEATURE_MODELS", {}).values():
            mp = feature.get("model_path")
            if not mp:
                continue
            feature_models.append(mp)
            if isinstance(mp, str) and mp.endswith(".tflite"):
                feature_models.append(mp.replace(".tflite", ".onnx"))

        for fp in feature_models:
            if not fp or not os.path.isfile(fp):
                wanted_present = False
                break

        for wake_id in OPENWAKEWORD_WAKEWORDS_TO_DOWNLOAD:
            p_t = id_to_tflite.get(wake_id)
            p_o = id_to_onnx.get(wake_id)
            if not p_t or not os.path.isfile(p_t):
                wanted_present = False
                break
            if not p_o or not os.path.isfile(p_o):
                wanted_present = False
                break

        if wanted_present:
            self.status_message.emit("OpenWakeWord models already present.")
            return

        # Preflight writability: create + remove a probe file.
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            probe_path = target_dir / ".qube_write_probe"
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write("probe")
            try:
                probe_path.unlink()
            except OSError:
                pass
        except Exception as exc:
            raise PermissionError(
                "OpenWakeWord models directory is not writable in this install. "
                f"Directory: {target_dir}"
            ) from exc

        self.status_message.emit("Downloading OpenWakeWord wakeword models…")

        openwakeword.utils.download_models(
            model_names=list(OPENWAKEWORD_WAKEWORDS_TO_DOWNLOAD),
            target_directory=str(target_dir),
        )

    def _download_community(self) -> None:
        self.status_message.emit("Downloading community wakewords…")

        dest_root = models_root() / "wakeword"
        dest_en = dest_root / "en"
        dest_root.mkdir(parents=True, exist_ok=True)
        dest_en.mkdir(parents=True, exist_ok=True)

        zip_url = COMMUNITY_REPO_ZIP_MAIN
        zip_path = dest_root / "community_wakewords_main.zip"
        part_path = Path(str(zip_path) + ".part")

        try:
            with requests.get(zip_url, stream=True, timeout=(30, 300)) as resp:
                resp.raise_for_status()

                # Stream zip to disk first, then extract deterministically.
                with open(part_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            f.write(chunk)

            os.replace(part_path, zip_path)
            self.status_message.emit("Extracting community models…")

            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.infolist():
                    name = member.filename.replace("\\", "/")

                    # Only extract files under */en/ and only ONNX/TFLite wakeword files.
                    if "/en/" not in name:
                        continue
                    if not (name.endswith(".onnx") or name.endswith(".tflite")):
                        continue

                    parts = name.split("/")
                    # Find the "en" directory component and drop everything before it.
                    try:
                        en_idx = parts.index("en")
                    except ValueError:
                        continue

                    rel_parts = parts[en_idx + 1 :]
                    if not rel_parts:
                        continue

                    # Filter out explicit community wakewords.
                    folder_norm = rel_parts[0].replace(" ", "_").replace("-", "_").lower().strip()
                    file_stem_norm = Path(rel_parts[-1]).stem.split("_v")[0].replace(" ", "_").replace("-", "_").lower().strip()
                    if folder_norm in COMMUNITY_WAKEWORDS_TO_EXCLUDE or file_stem_norm in COMMUNITY_WAKEWORDS_TO_EXCLUDE:
                        continue

                    out_path = dest_en.joinpath(*rel_parts)
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                        while True:
                            buf = src.read(1024 * 512)
                            if not buf:
                                break
                            dst.write(buf)
        finally:
            # Keep the zip as a cache if the user re-runs the download.
            # Do not delete zip_path to avoid re-downloading on retry.
            if part_path.exists():
                try:
                    part_path.unlink()
                except OSError:
                    pass

