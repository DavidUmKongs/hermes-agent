#!/usr/bin/env python3
"""Simple safe print that tries rich, falls back to regular print.

When rich is unavailable, any rich-style markup like ``[bold red]...[/bold red]``
is stripped from string arguments so the plain output stays readable instead of
leaking literal tags.
"""

import re

try:
    from rich import print as rich_print
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Matches rich markup tags like ``[bold red]``, ``[/bold red]``, ``[/]``, etc.
# Conservative: only strips bracketed tokens that look like style directives
# (letters, digits, slashes, spaces, # for hex colors).
_RICH_MARKUP_RE = re.compile(r"\[/?[a-zA-Z0-9 #_/-]*\]")


def _strip_markup(arg):
    if isinstance(arg, str):
        return _RICH_MARKUP_RE.sub("", arg)
    return arg


def safe_print(*args, **kwargs):
    """Try rich.print, fall back to regular print (with markup stripped)."""
    if RICH_AVAILABLE:
        try:
            rich_print(*args, **kwargs)
            return
        except Exception:
            pass
    # Fallback: strip rich markup so we don't print literal "[bold red]..." tags
    print(*(_strip_markup(a) for a in args), **kwargs)
