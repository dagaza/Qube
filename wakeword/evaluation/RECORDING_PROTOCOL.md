# Evaluation recording protocol

How to capture the held-out corpus so model comparisons are fair and reproducible.

## Coverage targets

- **≥ 4 accents**, **both genders**, **≥ 3 microphones** (laptop, USB, headset).
- **Quiet** and **noisy** environments (TV / music / background chatter).
- **Near-field** (~30 cm) and **far-field** (~2 m).

## Per speaker

1. **Positives** — say the wake phrase ~20× with natural variation (normal, fast,
   soft, slightly different intonation). One utterance per file.
2. **Adversarial** — say each sound-alike from the config's `adversarial_phrases`
   ~5× (e.g. "hey cube", "a cube").
3. **Long-form negative** — 5–10 min of natural speech / reading that never contains
   the wake word (feeds false-positives-per-hour).

## Format

- **16 kHz, mono, 16-bit PCM WAV** (matches Qube's audio worker).
- Convert if needed: `ffmpeg -i in.ext -ar 16000 -ac 1 out.wav`.

## Privacy

Real voices are sensitive. **Do not commit raw audio.** Keep only
`evaluation/corpus.json` (the index) in git; store audio out-of-band with consent
from each speaker.

## Consistency

Use the same files to evaluate **every** candidate model so the variant comparison
(Section 4) and threshold sweep are apples-to-apples.
