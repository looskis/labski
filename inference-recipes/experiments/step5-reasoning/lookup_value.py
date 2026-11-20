"""Extract a single six-digit value for a known lookup key from a stopped reply."""

import re


def lookup_value(key, text, finish):
    if finish != "stop" or not re.fullmatch(r"bin-[0-9]{4}", key):
        return None
    match = re.fullmatch(r"(?:" + re.escape(key) + r" = )?([0-9]{6})", text.strip())
    return match.group(1) if match else None
