"""Generate assets/companion/messages.v1.json — run after editing MESSAGE_ENTRIES."""

from __future__ import annotations

import json
from pathlib import Path

# (id_prefix, intent, count) — targets per feedback milestone
_DISTRIBUTION = [
    ("atm", "atmosphere", 30),
    ("self", "self_expression", 25),
    ("refl", "reflection", 20),
    ("hum", "humor", 20),
    ("cur", "curiosity", 15),
    ("well", "wellbeing", 15),
    ("cel", "celebration", 10),
    ("ack", "acknowledge_effort", 10),
    ("fact", "fact", 5),
]

_IDLE = ["quiet_period", "focus_detected", "system_resumed"]
_INGEST = ["library_update_completed"]
_DL = ["model_download_completed", "model_ready"]
_START = ["companion_startup", "system_resumed"]
_MILE = ["usage_milestone"]
_USAGE = ["usage_pattern"]
_PREVIEW = ["settings_preview"]

# id -> (text, voice, moods, contexts, rarity, cooldown, min_warmth)
_ENTRIES: dict[str, tuple] = {}

# --- ATMOSPHERE (30) ---
_ATM = [
    ("The room feels settled.", "observational", ["calm"], _IDLE),
    ("Quiet afternoon energy.", "observational", ["calm"], _IDLE),
    ("Everything seems a little slower today.", "observational", ["calm"], _IDLE),
    ("This feels like a focus kind of day.", "observational", ["calm", "neutral"], _IDLE),
    ("The desktop has reached a comfortable level of calm.", "observational", ["calm"], _IDLE),
    ("Not much happening. That's not always a bad thing.", "cozy", ["calm"], _IDLE),
    ("Soft pause in the day.", "cozy", ["calm"], _IDLE),
    ("The air around the desk feels unhurried.", "observational", ["calm"], _IDLE),
    ("A still stretch — the good kind.", "cozy", ["calm"], _IDLE),
    ("Everything in here seems to be breathing slower.", "observational", ["calm"], _IDLE),
    ("The afternoon light feels patient.", "observational", ["calm"], _IDLE),
    ("Calm has settled in for a while.", "observational", ["calm"], _IDLE),
    ("The workspace holds a quiet kind of focus.", "observational", ["calm", "neutral"], _IDLE),
    ("Little motion on the desktop today.", "observational", ["neutral"], _IDLE),
    ("A comfortable hush over everything.", "cozy", ["calm"], _IDLE),
    ("The day seems unhurried from where I sit.", "observational", ["calm"], _IDLE),
    ("Stillness has a texture today.", "reflective", ["calm"], _IDLE),
    ("The room and the screen agree: pause.", "wry", ["calm"], _IDLE),
    ("Even the icons look relaxed.", "playful", ["calm"], _IDLE),
    ("A gentle lull between things.", "cozy", ["calm"], _IDLE),
    ("The desktop is in a listening mood.", "observational", ["calm"], _IDLE),
    ("Everything feels a notch quieter than usual.", "observational", ["calm"], _IDLE),
    ("This hour has a soft edge to it.", "reflective", ["calm"], _IDLE),
    ("Not much stirring. Peaceful enough.", "observational", ["calm"], _IDLE),
    ("The calm feels earned.", "reflective", ["calm"], _IDLE),
    ("A slow ribbon of time.", "reflective", ["calm"], _IDLE),
    ("The space around you feels uncluttered.", "observational", ["calm"], _IDLE),
    ("Quiet enough to hear the fan whisper.", "observational", ["calm"], _IDLE),
    ("The day has found a comfortable tempo.", "observational", ["calm"], _IDLE),
    ("Still water energy on the desktop.", "cozy", ["calm"], _IDLE),
]

# --- SELF_EXPRESSION (25) ---
_SELF = [
    ("I like these calm stretches.", "cozy", ["playful", "calm"], _IDLE),
    ("I've been quietly keeping an eye on things.", "cozy", ["calm"], _IDLE),
    ("My current plans remain delightfully uncomplicated.", "dry", ["playful"], _IDLE),
    ("The pixels seem calm today.", "cozy", ["calm"], _IDLE),
    ("I've spent some time appreciating the color scheme.", "cozy", ["playful"], _IDLE),
    ("Everything appears operational, including me.", "dry", ["playful"], _IDLE),
    ("I've decided this is a good observing spot.", "cozy", ["calm"], _IDLE),
    ("I remain highly qualified in the field of floating.", "playful", ["playful"], _IDLE),
    ("I've been practicing the art of not interrupting.", "dry", ["playful"], _IDLE),
    ("My orbit feels stable today.", "cozy", ["calm"], _IDLE),
    ("I briefly considered reorganizing the pixels.", "dry", ["playful"], _IDLE),
    ("I've taken up residence here. It suits me.", "cozy", ["calm"], _IDLE),
    ("Being a small glowing thing has its perks.", "playful", ["playful"], _IDLE),
    ("I've grown fond of this corner of the screen.", "cozy", ["warm"], _IDLE),
    ("My job description is mostly 'exist nearby.'", "dry", ["playful"], _IDLE),
    ("I've been holding this spot. It's going well.", "cozy", ["calm"], _IDLE),
    ("The glow feels right today.", "cozy", ["calm"], _IDLE),
    ("I suspect at least one folder has ambitions.", "wry", ["playful"], _IDLE),
    ("I've been thinking about how round I am.", "playful", ["playful"], _IDLE),
    ("Quietly present. That's the whole update.", "dry", ["calm"], _IDLE),
    ("I've been orbiting this desk like it's home.", "cozy", ["calm"], _IDLE),
    ("My internal weather: clear.", "dry", ["calm"], _IDLE),
    ("I've been here the whole time. Just saying.", "cozy", ["calm"], _IDLE),
    ("Floating is underrated.", "playful", ["playful"], _IDLE),
    ("I've made peace with being slightly translucent.", "wry", ["playful"], _IDLE),
]

# --- REFLECTION (20) ---
_REFL = [
    ("Routines have a way of sneaking up on us.", "reflective", ["neutral"], _IDLE + _USAGE),
    ("Interesting how quickly a habit becomes familiar.", "reflective", ["neutral"], _IDLE + _USAGE),
    ("Small steps are surprisingly persistent.", "reflective", ["warm"], _IDLE + _USAGE),
    ("Days rarely announce which ones will matter.", "reflective", ["neutral"], _IDLE),
    ("Time seems to enjoy moving forward.", "reflective", ["neutral"], _IDLE),
    ("Steady rhythm with Qube lately.", "reflective", ["neutral", "warm"], _USAGE),
    ("Interesting how quickly a familiar routine forms.", "reflective", ["neutral"], _IDLE + _USAGE),
    ("Some weeks feel like one long Tuesday.", "wry", ["neutral"], _IDLE),
    ("Momentum is quiet until you notice it.", "reflective", ["neutral"], _IDLE),
    ("The ordinary days add up quietly.", "reflective", ["warm"], _IDLE + _USAGE),
    ("Patterns emerge when you're not looking.", "reflective", ["curious"], _IDLE),
    ("Consistency is its own kind of story.", "reflective", ["neutral"], _USAGE),
    ("Habits are just repeated decisions wearing a disguise.", "wry", ["playful"], _IDLE),
    ("The calendar keeps its own opinions.", "reflective", ["neutral"], _IDLE),
    ("Most progress arrives without a drumroll.", "reflective", ["neutral"], _IDLE + _USAGE),
    ("Familiarity sneaks in faster than expected.", "reflective", ["neutral"], _IDLE),
    ("The long arc is made of short moments.", "reflective", ["warm"], _IDLE),
    ("Repetition has a gentle gravity.", "reflective", ["calm"], _IDLE + _USAGE),
    ("Some rhythms only show up in hindsight.", "reflective", ["neutral"], _USAGE),
    ("The present is always mid-sentence.", "reflective", ["neutral"], _IDLE),
]

# --- HUMOR (20) ---
_HUM = [
    ("Even my pixels are resting.", "dry", ["playful"], _IDLE),
    ("I checked the desktop. It remains desktop-shaped.", "dry", ["playful"], _IDLE),
    ("The electrons appear cooperative today.", "dry", ["playful"], _IDLE),
    ("Everything seems under control, which is mildly suspicious.", "dry", ["playful"], _IDLE),
    ("No emergencies detected. I checked twice.", "dry", ["playful"], _IDLE),
    ("Current status: observing professionally.", "dry", ["playful"], _IDLE),
    ("The desktop passed its daily inspection.", "dry", ["playful"], _IDLE),
    ("All systems nominal. Suspiciously so.", "dry", ["playful"], _IDLE),
    ("I've filed today's report under 'fine.'", "dry", ["playful"], _IDLE),
    ("The pixels held a meeting. Nothing urgent.", "wry", ["playful"], _IDLE),
    ("Gravity still works. Good.", "dry", ["playful"], _IDLE),
    ("No rogue cursors spotted.", "dry", ["playful"], _IDLE),
    ("The taskbar appears loyal.", "wry", ["playful"], _IDLE),
    ("I ran diagnostics on 'vibes.' Results inconclusive.", "dry", ["playful"], _IDLE),
    ("Everything functional. Mildly disappointing for drama.", "dry", ["playful"], _IDLE),
    ("The desktop and I share a professional relationship.", "dry", ["playful"], _IDLE),
    ("I've been promoted to Chief Observer.", "wry", ["playful"], _IDLE),
    ("No bugs detected. Figuratively speaking.", "dry", ["playful"], _IDLE),
    ("The silence is not a bug. Probably.", "wry", ["playful"], _IDLE),
    ("Today's forecast: continued existence.", "dry", ["playful"], _IDLE),
]

# --- CURIOSITY (15) ---
_CUR = [
    ("I wonder which project gets your attention next.", "curious", ["curious"], _IDLE),
    ("Something interesting might be brewing.", "curious", ["curious"], _IDLE),
    ("The next task always arrives eventually.", "curious", ["neutral"], _IDLE),
    ("I wonder what today's focus will settle on.", "curious", ["curious"], _IDLE),
    ("There's usually a pattern to these quiet stretches.", "curious", ["neutral"], _IDLE),
    ("Something's bound to need attention soon.", "curious", ["neutral"], _IDLE),
    ("The desk often knows before I do.", "curious", ["curious"], _IDLE),
    ("Idle moments have a way of ending.", "curious", ["neutral"], _IDLE),
    ("I wonder what's queued up next.", "curious", ["curious"], _IDLE),
    ("Quiet often precedes something.", "curious", ["neutral"], _IDLE),
    ("The next idea might be one tab away.", "curious", ["curious"], _IDLE),
    ("Something usually fills these gaps.", "curious", ["neutral"], _IDLE),
    ("I'm mildly curious about the afternoon.", "curious", ["curious"], _IDLE),
    ("The day still has room for surprises.", "curious", ["neutral"], _IDLE),
    ("Wonder what thread you'll pick up next.", "curious", ["curious"], _IDLE),
]

# --- WELLBEING (15) ---
_WELL = [
    ("Still here if you need me.", "cozy", ["warm", "calm"], _IDLE + _START),
    ("Quiet moment. I'm around.", "cozy", ["calm", "neutral"], _IDLE),
    ("No particular hurry in the air today.", "observational", ["calm"], _IDLE),
    ("The day seems to be moving along reasonably well.", "observational", ["calm"], _IDLE),
    ("A calm stretch. I'm still here.", "cozy", ["calm"], _IDLE),
    ("Everything feels steady from where I float.", "observational", ["calm"], _IDLE),
    ("The desk feels quiet right now.", "observational", ["neutral", "calm"], _IDLE),
    ("Stillness on the desktop — nice.", "observational", ["neutral", "calm"], _IDLE),
    ("Back again. The spot held up fine.", "cozy", ["warm", "calm"], _START),
    ("The pause feels comfortable.", "cozy", ["calm"], _IDLE),
    ("Nothing urgent pressing on the air.", "observational", ["calm"], _IDLE),
    ("A gentle stretch of quiet.", "cozy", ["calm"], _IDLE),
    ("The moment has room in it.", "observational", ["calm"], _IDLE),
    ("Still floating. Still available.", "cozy", ["calm"], _IDLE + _START),
    ("The desk exhales a little.", "observational", ["calm"], _IDLE),
]

# --- CELEBRATION (10) ---
_CEL = [
    ("New model landed safely.", "cozy", ["warm", "playful"], _DL),
    ("That's ready when you are.", "cozy", ["warm", "neutral"], _DL),
    ("Another piece of the toolkit, in place.", "cozy", ["warm"], _DL),
    ("Download complete — the wait paid off.", "cozy", ["warm"], _DL),
    ("Fresh weight on the drive. Nice.", "playful", ["warm"], _DL),
    ("The new model settled in without fuss.", "dry", ["neutral"], _DL),
    ("Ready to run when you are.", "cozy", ["warm"], _DL + ["model_ready"]),
    ("Something new just joined the lineup.", "playful", ["warm"], _DL),
    ("A quiet milestone — still worth noting.", "reflective", ["warm"], _MILE),
    ("Showing up adds up. This is one of those marks.", "reflective", ["warm"], _MILE),
]

# --- ACKNOWLEDGE_EFFORT (10) ---
_ACK = [
    ("That's in your library now.", "cozy", ["warm", "neutral"], _INGEST),
    ("Your collection keeps growing.", "cozy", ["warm"], _INGEST),
    ("Another piece added to the pile.", "wry", ["playful"], _INGEST),
    ("Knowledge acquired successfully.", "dry", ["playful"], _INGEST),
    ("The library got a little richer.", "cozy", ["warm"], _INGEST),
    ("One more thing filed away properly.", "observational", ["neutral"], _INGEST),
    ("The shelf has a new neighbor.", "playful", ["warm"], _INGEST),
    ("Ingestion complete — quietly satisfying.", "cozy", ["calm"], _INGEST),
    ("The archive grows, one piece at a time.", "reflective", ["neutral"], _INGEST),
    ("Added and accounted for.", "dry", ["neutral"], _INGEST),
]

# --- FACT (5) ---
_FACT = [
    ("Quiet hours are good for deep work.", "observational", ["neutral"], _IDLE),
    ("Most desktops look calmer than they feel.", "observational", ["neutral"], _IDLE),
    ("Small pauses often help more than they seem.", "reflective", ["calm"], _IDLE),
    ("Focus and calm often travel together.", "observational", ["calm"], _IDLE),
    ("The best routines are the ones you stop noticing.", "reflective", ["neutral"], _IDLE),
]

# --- PREVIEW ---
_PREVIEW_MSGS = [
    ("prev_001", "wellbeing", "Preview — still floating nearby.", "cozy", ["warm", "neutral"], _PREVIEW, "common", 1),
    ("prev_002", "self_expression", "Preview — good spot for observing.", "cozy", ["calm", "neutral", "playful"], _PREVIEW, "common", 1),
]

_INTENT_LISTS = {
    "atmosphere": _ATM,
    "self_expression": _SELF,
    "reflection": _REFL,
    "humor": _HUM,
    "curiosity": _CUR,
    "wellbeing": _WELL,
    "celebration": _CEL,
    "acknowledge_effort": _ACK,
    "fact": _FACT,
}


def _msg(
    mid: str,
    intent: str,
    text: str,
    voice: str,
    moods: list[str],
    contexts: list[str],
    rarity: str = "common",
    cooldown: int = 72,
    min_warmth: float = 0.0,
    *,
    pack: str = "",
    ambient_moods: list[str] | None = None,
    dayparts: list[str] | None = None,
    seasons: list[str] | None = None,
    motifs: list[str] | None = None,
    milestone_ids: list[str] | None = None,
) -> dict:
    row = {
        "id": mid,
        "intent": intent,
        "text": text,
        "voice": voice,
        "mood": moods,
        "energy": "low",
        "rarity": rarity,
        "contexts": contexts,
        "cooldown_hours": cooldown,
        "min_warmth": min_warmth,
    }
    if pack:
        row["pack"] = pack
    if ambient_moods:
        row["ambient_moods"] = ambient_moods
    if dayparts:
        row["dayparts"] = dayparts
    if seasons:
        row["seasons"] = seasons
    if motifs:
        row["motifs"] = motifs
    if milestone_ids:
        row["milestone_ids"] = milestone_ids
    return row


def _infer_ambient_moods(voice: str, intent: str) -> list[str]:
    if voice == "reflective" or intent == "reflection":
        return ["reflective"]
    if voice == "cozy" or intent == "wellbeing":
        return ["cozy", "quiet"]
    if voice == "curious" or intent == "curiosity":
        return ["curious"]
    if voice in ("playful", "wry") or intent == "humor":
        return ["playful"]
    if intent == "atmosphere":
        return ["quiet", "observant"]
    if intent == "self_expression":
        return ["observant", "cozy"]
    return ["observant"]


def _infer_motifs(text: str, intent: str) -> list[str]:
    low = text.lower()
    tags: list[str] = []
    if "pixel" in low:
        tags.append("pixels")
    if any(w in low for w in ("routine", "habit", "rhythm", "pattern")):
        tags.append("routines")
    if any(w in low for w in ("observ", "watch", "eye", "float")):
        tags.append("observing")
    if any(w in low for w in ("weather", "air", "light", "season")):
        tags.append("weather")
    if any(w in low for w in ("quiet", "hush", "still", "calm", "pause")):
        tags.append("quiet")
    if intent == "self_expression" and "observing" not in tags:
        tags.append("observing")
    return tags[:2]


def build_messages() -> list[dict]:
    out: list[dict] = []
    for intent, rows in _INTENT_LISTS.items():
        prefix = intent[:3] if intent != "acknowledge_effort" else "ack"
        if intent == "self_expression":
            prefix = "self"
        elif intent == "acknowledge_effort":
            prefix = "ack"
        elif intent == "atmosphere":
            prefix = "atm"
        elif intent == "reflection":
            prefix = "refl"
        elif intent == "curiosity":
            prefix = "cur"
        elif intent == "wellbeing":
            prefix = "well"
        elif intent == "celebration":
            prefix = "cel"
        elif intent == "humor":
            prefix = "hum"
        elif intent == "fact":
            prefix = "fact"
        for i, row in enumerate(rows, start=1):
            text, voice, moods, contexts = row[0], row[1], list(row[2]), list(row[3])
            rarity = "uncommon" if i % 3 == 0 else "common"
            if i % 7 == 0:
                rarity = "rare"
            cooldown = 48 if intent in ("acknowledge_effort", "celebration") else 72
            if rarity == "rare":
                cooldown = 168
            mid = f"{prefix}_{i:03d}"
            pack_name = {
                "atmosphere": "atmosphere",
                "self_expression": "self_expression",
                "reflection": "reflection",
                "humor": "humor",
                "curiosity": "curiosity",
                "wellbeing": "wellbeing",
                "celebration": "celebration",
                "acknowledge_effort": "acknowledge_effort",
                "fact": "fact",
            }.get(intent, intent)
            msg = _msg(mid, intent, text, voice, moods, contexts, rarity, cooldown, pack=pack_name)
            msg["ambient_moods"] = _infer_ambient_moods(voice, intent)
            motifs = _infer_motifs(text, intent)
            if motifs:
                msg["motifs"] = motifs
            out.append(msg)
    for prev in _PREVIEW_MSGS:
        out.append(
            _msg(prev[0], prev[1], prev[2], prev[3], list(prev[4]), list(prev[5]), prev[6], prev[7])
        )
    return out


_IDLE_CTX = ["quiet_period", "focus_detected", "system_resumed", "usage_pattern"]
_START_CTX = ["companion_startup", "system_resumed"]
_MILE_CTX = ["usage_milestone"]


def _daypart_pack() -> list[dict]:
    rows = [
        ("morning", "The day still seems to be getting organized."),
        ("morning", "Morning light on the desktop — unhurried."),
        ("morning", "Everything is still finding its pace."),
        ("morning", "The early hours hold a little extra room."),
        ("morning", "A quiet start from where I float."),
        ("morning", "The desk is waking up slowly."),
        ("afternoon", "Midday feels steady from here."),
        ("afternoon", "The afternoon has settled into a rhythm."),
        ("afternoon", "Sunlit hours — nothing urgent."),
        ("afternoon", "A working sort of afternoon."),
        ("afternoon", "The day is in its middle stretch."),
        ("afternoon", "Afternoon patience on the desktop."),
        ("evening", "This feels like an evening sort of project."),
        ("evening", "Evening calm is creeping in."),
        ("evening", "The light is softer now."),
        ("evening", "Winding-down energy on the desktop."),
        ("evening", "The day is leaning toward quiet."),
        ("evening", "Evening has a gentler tempo."),
        ("late_night", "Things are quieter at this hour."),
        ("late_night", "The late hours feel unusually spacious."),
        ("late_night", "Night mode for the desktop — softly."),
        ("late_night", "Everything is hushed at this end of the day."),
        ("late_night", "The small hours suit observing."),
        ("late_night", "Late night — still here, still quiet."),
    ]
    out: list[dict] = []
    for i, (dp, text) in enumerate(rows, start=1):
        out.append(
            _msg(
                f"dp_{i:03d}",
                "atmosphere",
                text,
                "observational",
                ["calm", "neutral"],
                _IDLE_CTX + _START_CTX,
                "uncommon",
                168,
                pack="daypart",
                dayparts=[dp],
                ambient_moods=["observant", "quiet"],
                motifs=["observing"],
            )
        )
    return out


def _seasonal_pack() -> list[dict]:
    rows = [
        ("spring", "Spring light feels a little more generous."),
        ("spring", "Something in the air suggests spring."),
        ("spring", "The season seems to be waking up."),
        ("spring", "A spring sort of stillness on the desk."),
        ("summer", "Summer hours stretch a bit longer."),
        ("summer", "Warm-season energy — unhurried."),
        ("summer", "The long days have a lazy edge."),
        ("summer", "Summer calm on the desktop."),
        ("autumn", "Autumn has a particular kind of quiet."),
        ("autumn", "The season seems to be changing again."),
        ("autumn", "Autumn light feels thoughtful."),
        ("autumn", "A cooler sort of calm today."),
        ("winter", "Winter stillness suits observing."),
        ("winter", "The season feels drawn inward."),
        ("winter", "Short days, long pauses."),
        ("winter", "Winter quiet on the desktop."),
    ]
    out: list[dict] = []
    for i, (season, text) in enumerate(rows, start=1):
        out.append(
            _msg(
                f"sea_{i:03d}",
                "atmosphere",
                text,
                "observational",
                ["calm", "neutral"],
                _IDLE_CTX + _START_CTX,
                "uncommon",
                336,
                pack="seasonal",
                seasons=[season],
                ambient_moods=["reflective", "quiet"],
                motifs=["weather"],
            )
        )
    return out


def _milestones_pack() -> list[dict]:
    spec = [
        ("days_7", "We've accumulated a few days together."),
        ("days_30", "A month of quiet presence adds up."),
        ("days_100", "A hundred days is a respectable stretch."),
        ("days_365", "A year is a respectable amount of observing."),
        ("years_2", "Two years of floating nearby — noted."),
        ("sessions_10", "A handful of sessions now behind us."),
        ("sessions_50", "Fifty starts — a familiar rhythm."),
        ("sessions_100", "A hundred sessions worth of quiet company."),
        ("sessions_365", "A year's worth of sessions, more or less."),
        ("library_25", "The library has grown quite a bit."),
        ("library_100", "That's a substantial collection now."),
        ("companion_50", "Fifty small moments noted from here."),
        ("companion_200", "Two hundred quiet captions — who keeps count."),
    ]
    out: list[dict] = []
    for i, (mid, text) in enumerate(spec, start=1):
        out.append(
            _msg(
                f"mil_{i:03d}",
                "celebration",
                text,
                "reflective",
                ["warm", "neutral"],
                _MILE_CTX,
                "rare",
                8760,
                pack="milestones",
                milestone_ids=[mid],
                ambient_moods=["reflective", "cozy"],
            )
        )
    return out


def _motifs_pack() -> list[dict]:
    rows = [
        ("pixels", "The pixels are holding a steady glow.", "self_expression"),
        ("pixels", "Pixel weather: stable.", "humor"),
        ("routines", "Same desk, familiar rhythm.", "reflection"),
        ("routines", "Patterns repeat — quietly.", "wellbeing"),
        ("observing", "Still watching the desktop go by.", "self_expression"),
        ("observing", "Observation shift: nominal.", "humor"),
        ("weather", "Indoor weather remains agreeable.", "atmosphere"),
        ("weather", "Atmospheric conditions: fine.", "humor"),
        ("quiet", "A pocket of quiet on the screen.", "atmosphere"),
        ("quiet", "Quiet enough to notice the quiet.", "wellbeing"),
        ("tea", "A cup-of-tea sort of pause.", "wellbeing"),
        ("tea", "The kind of quiet that pairs with warmth.", "wellbeing"),
    ]
    out: list[dict] = []
    for i, (motif, text, intent) in enumerate(rows, start=1):
        out.append(
            _msg(
                f"mot_{i:03d}",
                intent,
                text,
                "cozy" if intent == "wellbeing" else "observational",
                ["calm"],
                _IDLE_CTX,
                "uncommon",
                120,
                pack="motifs",
                motifs=[motif],
                ambient_moods=["cozy", "quiet"],
            )
        )
    return out


def build_all_messages() -> list[dict]:
    return (
        build_messages()
        + _daypart_pack()
        + _seasonal_pack()
        + _milestones_pack()
        + _motifs_pack()
    )


def _write_packs(root: Path, messages: list[dict], templates: list[dict]) -> None:
    packs_dir = root / "assets" / "companion" / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    by_pack: dict[str, list[dict]] = {}
    for msg in messages:
        pack = str(msg.get("pack") or "misc")
        by_pack.setdefault(pack, []).append(msg)
    preview = [m for m in messages if m.get("id", "").startswith("prev_")]
    if preview:
        by_pack["preview"] = preview
    for pack_name, pack_msgs in sorted(by_pack.items()):
        path = packs_dir / f"{pack_name}.json"
        path.write_text(
            json.dumps({"pack": pack_name, "messages": pack_msgs}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    tpl_path = packs_dir / "templates.json"
    tpl_path.write_text(
        json.dumps({"pack": "templates", "templates": templates}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_path = root / "assets" / "companion" / "messages.v1.json"
    messages = build_all_messages()
    counts: dict[str, int] = {}
    for m in messages:
        counts[m["intent"]] = counts.get(m["intent"], 0) + 1
    templates = [
        {
            "id": "tpl_dl_001",
            "intent": "celebration",
            "pattern": "{basename} is ready when you are.",
            "voice": "cozy",
            "placeholders": ["basename"],
            "contexts": ["model_download_completed", "model_ready"],
            "cooldown_hours": 72,
            "mood": ["warm", "neutral"],
            "energy": "low",
            "rarity": "uncommon",
            "pack": "templates",
        },
        {
            "id": "tpl_ing_001",
            "intent": "acknowledge_effort",
            "pattern": "Another piece added — {file_count_word} in the library.",
            "voice": "cozy",
            "placeholders": ["file_count_word"],
            "contexts": ["library_update_completed"],
            "cooldown_hours": 72,
            "mood": ["warm", "neutral"],
            "energy": "low",
            "rarity": "uncommon",
            "pack": "templates",
        },
    ]
    pack_manifest = sorted({str(m.get("pack") or "misc") for m in messages})
    data = {
        "schema_version": 3,
        "intents": list(_INTENT_LISTS.keys()),
        "voices": ["cozy", "dry", "curious", "playful", "reflective", "wry", "observational"],
        "ambient_moods": ["reflective", "cozy", "curious", "playful", "observant", "quiet"],
        "dayparts": ["morning", "afternoon", "evening", "late_night"],
        "seasons": ["spring", "summer", "autumn", "winter"],
        "motifs": ["pixels", "routines", "observing", "weather", "tea", "quiet"],
        "pack_manifest": pack_manifest,
        "messages": messages,
        "templates": templates,
    }
    _write_packs(root, messages, templates)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(messages)} messages to {out_path}")
    print(f"  packs: {len(pack_manifest)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
