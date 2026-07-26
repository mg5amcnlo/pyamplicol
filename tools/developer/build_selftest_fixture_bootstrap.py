# SPDX-License-Identifier: 0BSD
"""Build the non-deployable candidate used to regenerate the self-test fixture."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import _pyamplicol_build as backend  # noqa: E402


def _workspace_output(raw: Path) -> Path:
    output = (raw if raw.is_absolute() else ROOT / raw).resolve()
    root = ROOT.resolve()
    if output == root or root not in output.parents:
        raise RuntimeError("bootstrap wheel directory must be inside the workspace")
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise RuntimeError("bootstrap wheel destination is not a safe directory")
        if any(output.iterdir()):
            raise RuntimeError("bootstrap wheel destination must be empty")
    else:
        output.mkdir(parents=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-directory",
        type=Path,
        default=Path(".artifacts/selftest-fixture-bootstrap"),
    )
    args = parser.parse_args(argv)
    output = _workspace_output(args.wheel_directory)
    filename = backend.build_selftest_fixture_bootstrap_wheel(str(output))
    wheel = output / filename
    if not wheel.is_file():
        raise RuntimeError("bootstrap build did not produce its declared wheel")
    print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
