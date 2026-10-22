#!/usr/bin/env python3
"""Strict local-link checker for the built MkDocs site."""
from __future__ import annotations
import argparse
import html.parser
import sys
import urllib.parse
from pathlib import Path


class Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.values.append(value)


def target_for(root: Path, current: Path, raw: str) -> Path | None:
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme or parsed.netloc or raw.startswith(("#", "mailto:", "javascript:")):
        return None
    path = urllib.parse.unquote(parsed.path)
    if not path:
        return None
    base = "/android-moe-engine"
    if path == base or path == base + "/":
        destination = root
    elif path.startswith(base + "/"):
        destination = root / path[len(base) + 1:]
    elif path.startswith("/"):
        destination = root / path.lstrip("/")
    else:
        destination = current.parent / path
    if destination.is_dir() or path.endswith("/"):
        destination = destination / "index.html"
    return destination.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path("site"))
    args = parser.parse_args()
    root = args.site.resolve()
    if not root.is_dir():
        print(f"missing built site: {root}", file=sys.stderr)
        return 2
    failures: list[str] = []
    pages = sorted(root.rglob("*.html"))
    if not pages:
        print("no HTML pages found", file=sys.stderr)
        return 2
    for page in pages:
        links = Links()
        links.feed(page.read_text(encoding="utf-8"))
        for raw in links.values:
            destination = target_for(root, page, raw)
            if destination is not None and not destination.is_file():
                failures.append(f"{page.relative_to(root)} -> {raw}")
    if failures:
        print("broken local links:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"checked {len(pages)} HTML pages; local links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())