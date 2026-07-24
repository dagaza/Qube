# Voice & Audio

## Common questions

- How do I choose my microphone and speakers?
- Where do I configure text-to-speech voice?
- How do I set up a wakeword?
- Why is Qube not hearing me?
- Why does my microphone level stay low in the meter?

## What it is

**Voice & Audio** settings control how Qube listens, speaks, and reacts to wake phrases. Pick **Audio Input** and **Audio Output**, turn **Enable Voice Input** and **Enable TTS Voice** on or off, choose **TTS Voice**, tune speech detection, optionally enable wakeword models, and pin controls to the Conversations toolbar.

Push-to-talk in **Conversations** is separate from always-on **Enable Voice Input** (available here and in the tools panel).

## Where to find it

Open **Settings → Voice & Audio** (settings section `voice.audio`). Press **?** on the page header for the guided tour (`settings.voice_audio`).

## Also called

audio settings, microphone settings, TTS voice, wakeword, speech input, AUDIO & HARDWARE

## How to…

1. **Pick devices** — Choose **Audio Input** and **Audio Output** from the dropdowns under **Devices** (`Select Input Device…`, `Select Output Device…`).
2. **Enable voice input** — Turn on **Enable Voice Input** under **Audio Input** (or use the same toggle in the Conversations tools panel) when you want always-on listening and wakeword detection.
3. **Check the level meter** — Speak while watching the small colored bar beside the microphone icon in the top bar. Use the **lightbulb hint** next to **Audio Input** to highlight it. See [Microphone level meter](#microphone-level-meter-top-bar) if one mic looks much quieter than another.
4. **Enable spoken replies** — Turn on **Enable TTS Voice**, then pick **TTS Voice** (`Select Voice…`). Download **Download base TTS model** if Kokoro is missing.
5. **Prepare speech recognition** — Download **Download base STT model** if Whisper Small is missing before using voice input.
6. **Select a wakeword** — Under **Wakeword**, choose **Active Wakeword** (`Select Wakeword…`). Download **Download OpenWakeWord models** or **Download Community models** as needed; test with **Open Wakeword Test Lab**.
7. **Tune capture** — Adjust **Silence Cutoff** and **VAD Threshold** under **Speech Detection**.
8. **Pin toolbar controls** — Use **Pin Audio Controls to Toolbar** and **Pin TTS Voice selector to Toolbar** to expose devices/voice beside chat.

## Microphone level meter (top bar)

When **Enable Voice Input** is on—or for a few seconds after you tap the **lightbulb hint** beside **Audio Input**—the colored bar next to the microphone icon in the top bar shows how loud your **selected microphone** sounds **to the computer**.

**Different microphones can look very different on this bar. That is normal.**

| What you might see | What it usually means |
|--------------------|------------------------|
| Bar fills up when you talk at normal volume | Common with **webcam**, **headset**, or **built-in** mics. They often boost sound automatically. |
| Bar only reaches part-way even when you speak up | Common with **USB mics**, **XLR mics**, or mics through an **audio interface** or **preamp**. The signal may be quieter in the computer even though it sounds fine in the room. |

**If one mic seems too quiet in Qube:**

1. Confirm you picked the right device in **Settings → Voice & Audio → Audio Input** (or the chevron beside the top-bar meter).
2. Open your **computer’s sound settings**, select the **same microphone**, and **raise its input volume**:
   - **Windows:** Settings → System → Sound → Input → choose your mic → adjust volume.
   - **macOS:** System Settings → Sound → Input → choose your mic → adjust input volume.
   - **Linux:** Settings → Sound → Input (or your desktop’s volume app, e.g. PulseAudio Volume Control).
3. If your mic or interface has a **gain** or **volume knob**, turn it up gradually while watching the bar. Aim for clear movement when you talk—not necessarily a full bar.
4. Tap the **lightbulb hint** next to **Audio Input** and speak to test; the bar pulses to draw your eye while you adjust.

**Good to know:** The bar does **not** need to hit 100% for voice to work. Mid-range movement is often healthy. If Qube **does not** pick up speech, raise OS input volume or lower **VAD Threshold** under **Speech Detection**. If the bar is **always** maxed out, your mic may be too loud in the computer—try lowering OS input volume for that device.

## Controls

<!-- include:generated/controls/voice-audio.md -->

## Related

- [Voice or microphone not working](../../troubleshooting/voice-or-microphone-not-working.md) — when input devices fail
- [Desktop Companion settings](desktop-companion.md) — companion voice and visibility
- [Settings sections reference](../../reference/settings-sections.md) — all settings pages
