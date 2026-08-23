"""Bridge huggingface_hub download progress to bootstrap UI callbacks."""

from __future__ import annotations

from collections.abc import Callable

from huggingface_hub.utils import tqdm as hf_tqdm

DownloadProgressCallback = Callable[[str, str, int, str], None]


def bootstrap_hub_tqdm_factory(
    on_progress: DownloadProgressCallback,
    *,
    step_label: str,
    filename: str,
    source_display: str,
    expected_total_bytes: int = 0,
) -> type[hf_tqdm]:
    """Build a tqdm subclass that forwards byte progress to ``on_progress``."""
    expected_total = max(0, int(expected_total_bytes))

    class _BootstrapHubTqdm(hf_tqdm):
        def update(self, n: float = 1) -> bool | None:
            result = super().update(n)
            total = int(self.total or 0) or expected_total
            if total > 0:
                pct = min(99, int(float(self.n) * 100 / total))
            elif self.n > 0:
                pct = min(99, int(float(self.n) // (1024 * 1024)))
            else:
                pct = 0
            on_progress(step_label, filename, pct, source_display)
            return result

    return _BootstrapHubTqdm
