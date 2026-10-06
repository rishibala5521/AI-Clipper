import math


def format_clock(seconds: float, *, round_up: bool = False) -> str:
    """Seconds -> 'M:SS', or 'H:MM:SS' from one hour up. Whole seconds only."""
    # The tiny 1e-9 guards against floating point noise such as 599.9999999.
    whole = math.ceil(seconds - 1e-9) if round_up else math.floor(seconds + 1e-9)
    whole = max(0, int(whole))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"