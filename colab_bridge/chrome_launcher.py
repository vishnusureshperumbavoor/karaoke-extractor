#!/usr/bin/env python3
"""BROWSER= helper for colab-mcp: open Colab URL in the selected Chrome profile."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colab_bridge.chrome_profiles import open_url_with_profile  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: chrome_launcher.py <url>", file=sys.stderr)
        return 2
    url = sys.argv[1]
    try:
        open_url_with_profile(url)
    except Exception as exc:
        print(f"[karaoke chrome_launcher] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
