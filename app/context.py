"""Context layer: normalizes user-supplied situational context and AUTO-DETECTS
Nigerian environmental signals where possible from the current date.

Auto-detection rules:
- salary_week: 23-30 of the month (Nigerian payroll convention)
- end_of_month: 27-31 of the month and explicit budget pressure
- festive: month is December (Christmas) or a few days before Eid (approx)
- time_bucket: derived from current hour if not supplied

User-supplied flags always take precedence over auto-detection.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

NIGERIAN_FLAGS = {
    "rainy": "Lagos rains amplify traffic and delivery delays.",
    "fuel_scarcity": "Fuel queues are pushing delivery costs and times up.",
    "salary_week": "It's salary week - users are more tolerant of price.",
    "end_of_month": "Money is tight - users are extra price-sensitive.",
    "festive": "Festive mood - users are more generous and forgiving.",
    "traffic_heavy": "Heavy traffic - delivery patience is low.",
    "power_outage": "PHCN issues today - frustration baseline is elevated.",
    "school_resumption": "Back-to-school spend is high; budgets are tighter.",
}


def _bucket_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late_night"


def _auto_flags(now: datetime) -> list[str]:
    flags = []
    day = now.day
    month = now.month
    # Salary week heuristic (Nigerian payroll: 25th-30th is common)
    if 23 <= day <= 30:
        flags.append("salary_week")
    if day >= 27:
        flags.append("end_of_month")
    # Festive: December everywhere; first week of January (carry-over)
    if month == 12 or (month == 1 and day <= 7):
        flags.append("festive")
    # Back to school: early September
    if month == 9 and day <= 14:
        flags.append("school_resumption")
    return flags


def normalize(raw: dict[str, Any] | None, *, now: datetime | None = None) -> dict[str, Any]:
    raw = dict(raw or {})
    now = now or datetime.now()

    time_str = raw.get("time")
    if isinstance(time_str, str) and time_str.lower() in {"morning", "afternoon", "evening", "night", "late_night"}:
        time_bucket = "late_night" if time_str.lower() == "night" else time_str.lower()
    else:
        time_bucket = _bucket_hour(now.hour)

    mood = raw.get("mood")
    weather = raw.get("weather")

    # Start with auto-detected flags, then layer user-supplied flags on top.
    flags: list[str] = list(_auto_flags(now))
    if weather and "rain" in str(weather).lower():
        flags.append("rainy")
    for key in ("fuel_scarcity", "salary_week", "end_of_month", "festive", "traffic_heavy", "power_outage", "school_resumption"):
        val = raw.get(key)
        if val is True:
            if key not in flags:
                flags.append(key)
        elif val is False:
            # user explicitly turned off an auto-flag
            flags = [f for f in flags if f != key]

    # Dedupe while preserving order
    seen = set()
    flags = [f for f in flags if not (f in seen or seen.add(f))]

    notes = [NIGERIAN_FLAGS[f] for f in flags if f in NIGERIAN_FLAGS]
    return {
        "time_bucket": time_bucket,
        "weather": weather,
        "mood_hint": mood,
        "nigerian_flags": flags,
        "context_notes": notes,
        "as_of": now.isoformat(),
    }
