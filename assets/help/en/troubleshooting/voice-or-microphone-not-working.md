# Voice or microphone not working

## Common questions

- Qube does not hear me—how do I fix the microphone?
- Speech recognition starts then stops immediately
- No audio output from text-to-speech
- My microphone level bar stays low even when I speak loudly

## What it is

Voice issues usually trace to wrong **Audio Input** or **Audio Output** devices, OS permission blocks, another app holding the microphone, **input volume set too low in the OS**, or wakeword sensitivity set too low/high. TTS problems often mean the wrong output device or muted system volume.

The top-bar **microphone level meter** shows how loud the selected mic is **to the computer**—not how loud you sound in the room. Webcam and headset mics often fill the bar; studio or interface mics may sit lower until you raise OS input volume or device gain. See [Microphone level meter](../features/settings/voice-audio.md#microphone-level-meter-top-bar) in Voice & Audio settings.

## Where to find it

Configure devices in **Settings → Voice & Audio**. Test in Conversations or the wakeword test lab after changes.

## Also called

mic not working, speech input failed, cannot hear Qube, TTS silent, wakeword not triggering

## How to…

1. Open **Settings → Voice & Audio** and select the correct **Audio Input** and **Audio Output** devices.
2. Confirm the OS grants microphone permission to Qube (system privacy settings).
3. Close other apps that may exclusive-lock the microphone (voice chat, DAWs, browsers).
4. **If the level meter barely moves:** open your **computer’s sound settings**, select the same microphone, and **raise input volume** (and any **gain knob** on the mic or interface). Use the **lightbulb hint** beside **Audio Input** in Qube while you test. See [Microphone level meter](../features/settings/voice-audio.md#microphone-level-meter-top-bar).
5. Adjust wakeword sensitivity or disable wakeword to test push-to-talk / manual voice turns.
6. If speech still does not start, lower **VAD Threshold** in **Settings → Voice & Audio** when your mic sends a quieter signal.
7. Raise system and app output volume; pick a different **Voice** for TTS if one voice fails.
8. Restart Qube after changing default system devices at the OS level.

## Related

- [Voice & Audio settings](../features/settings/voice-audio.md) — devices, TTS, wakeword
- [Desktop Companion settings](../features/settings/desktop-companion.md) — companion voice path
