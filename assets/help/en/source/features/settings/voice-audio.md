# Voice & Audio

## Common questions

- How do I choose my microphone and speakers?
- Where do I configure text-to-speech voice?
- How do I set up a wakeword?
- Why is Qube not hearing me?

## What it is

**Voice & Audio** settings control how Qube listens, speaks, and reacts to wake phrases. Pick **Audio Input** and **Audio Output**, configure **Enable TTS Voice** and **TTS Voice**, tune speech detection, optionally enable wakeword models, and pin controls to the Conversations toolbar.

Push-to-talk in **Conversations** is separate from always-on **Enable Voice Input** in the tools panel.

## Where to find it

Open **Settings → Voice & Audio** (settings section `voice.audio`). Press **?** on the page header for the guided tour (`settings.voice_audio`).

## Also called

audio settings, microphone settings, TTS voice, wakeword, speech input, AUDIO & HARDWARE

## How to…

1. **Pick devices** — Choose **Audio Input** and **Audio Output** from the dropdowns under **Devices** (`Select Input Device…`, `Select Output Device…`).
2. **Enable spoken replies** — Turn on **Enable TTS Voice**, then pick **TTS Voice** (`Select Voice…`). Download **Download base TTS model** if Kokoro is missing.
3. **Prepare speech recognition** — Download **Download base STT model** if Whisper Small is missing before using voice input.
4. **Select a wakeword** — Under **Wakeword**, choose **Active Wakeword** (`Select Wakeword…`). Download **Download OpenWakeWord models** or **Download Community models** as needed; test with **Open Wakeword Test Lab**.
5. **Tune capture** — Adjust **Silence Cutoff** and **VAD Threshold** under **Speech Detection**.
6. **Pin toolbar controls** — Use **Pin Audio Controls to Toolbar** and **Pin TTS Voice selector to Toolbar** to expose devices/voice beside chat.

## Controls

<!-- include:generated/controls/voice-audio.md -->

## Related

- [Voice or microphone not working](../../troubleshooting/voice-or-microphone-not-working.md) — when input devices fail
- [Desktop Companion settings](desktop-companion.md) — companion voice and visibility
- [Settings sections reference](../../reference/settings-sections.md) — all settings pages
