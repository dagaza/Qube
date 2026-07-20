# Voice or microphone not working

## Common questions

- Qube does not hear me—how do I fix the microphone?
- Speech recognition starts then stops immediately
- No audio output from text-to-speech

## What it is

Voice issues usually trace to wrong **Audio Input** or **Audio Output** devices, OS permission blocks, another app holding the microphone, or wakeword sensitivity set too low/high. TTS problems often mean the wrong output device or muted system volume.

## Where to find it

Configure devices in **Settings → Voice & Audio**. Test in Conversations or the wakeword test lab after changes.

## Also called

mic not working, speech input failed, cannot hear Qube, TTS silent, wakeword not triggering

## How to…

1. Open **Settings → Voice & Audio** and select the correct **Audio Input** and **Audio Output** devices.
2. Confirm the OS grants microphone permission to Qube (system privacy settings).
3. Close other apps that may exclusive-lock the microphone (voice chat, DAWs, browsers).
4. Adjust wakeword sensitivity or disable wakeword to test push-to-talk / manual voice turns.
5. Raise system and app output volume; pick a different **Voice** for TTS if one voice fails.
6. Restart Qube after changing default system devices at the OS level.

## Related

- [Voice & Audio settings](../features/settings/voice-audio.md) — devices, TTS, wakeword
- [Desktop Companion settings](../features/settings/desktop-companion.md) — companion voice path
