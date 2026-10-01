# vvv THOG allocate permanent readable host prefixes while durable UUIDs remain authoritative
"""Host prefix allocation shared by Networks and the Node Agent."""
import re
import secrets
import string


def choose_prefix(hostname, occupied=(), preferred=None):
    used = set(occupied)
    if isinstance(preferred, str) and re.fullmatch(r"[A-Z]{3}", preferred) and preferred not in used:
        return preferred
    prefix = re.sub(r"[^A-Za-z]", "", hostname).upper()[:3]
    if len(prefix) == 3 and prefix not in used:
        return prefix
    if len(used) >= 26 ** 3:
        raise RuntimeError("Three-letter host prefixes exhausted")
    while True:
        prefix = "".join(secrets.choice(string.ascii_uppercase) for _ in range(3))
        if prefix not in used:
            return prefix
# ^^^ THOG
