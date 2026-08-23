"""Timeframe conversion helpers."""

from datetime import timedelta


def timeframe_delta(timeframe: str) -> timedelta:
    """Convert a supported compact timeframe into a duration."""

    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    raise ValueError(f"Unsupported timeframe: {timeframe}")
