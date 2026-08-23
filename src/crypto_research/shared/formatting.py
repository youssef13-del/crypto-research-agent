"""Deterministic display formatting."""


def format_money(value: float) -> str:
    """Format a currency value with compact suffixes."""

    for threshold, suffix in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f} {suffix}"
    if abs(value) >= 100:
        return f"${value:,.0f}"
    return f"${value:,.4f}".rstrip("0").rstrip(".")


def format_compact_number(value: float, *, currency: bool = False) -> str:
    """Format a count with compact suffixes, optionally dollar-prefixed."""

    for threshold, suffix in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(value) >= threshold:
            return f"{'$' if currency else ''}{value / threshold:,.2f} {suffix}"
    return f"{'$' if currency else ''}{value:,.0f}"
