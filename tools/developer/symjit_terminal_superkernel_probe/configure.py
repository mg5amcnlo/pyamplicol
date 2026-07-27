#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Bind this disposable Cargo package to an explicit configured SymJIT tree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symjit-path",
        type=Path,
        default=None,
        help="configured pinned SymJIT checkout (defaults to SYMJIT_PATH)",
    )
    args = parser.parse_args()
    raw = args.symjit_path
    if raw is None:
        value = os.environ.get("SYMJIT_PATH")
        if value is None:
            parser.error("--symjit-path or SYMJIT_PATH is required")
        raw = Path(value)
    source = raw.expanduser().resolve(strict=True)
    if (
        not (source / "Cargo.toml").is_file()
        or not (source / "rust/direct.rs").is_file()
    ):
        parser.error(f"not a configured SymJIT source tree: {source}")

    link = Path(__file__).resolve().parent / ".symjit-path"
    if link.is_symlink() and link.resolve(strict=True) == source:
        print(link)
        return 0
    if link.exists() or link.is_symlink():
        parser.error(f"refusing to replace existing path: {link}")
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    temporary.symlink_to(source, target_is_directory=True)
    os.replace(temporary, link)
    print(link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
