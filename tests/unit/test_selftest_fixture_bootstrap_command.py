# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/developer/build_selftest_fixture_bootstrap.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_selftest_fixture_bootstrap",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_command_rejects_output_outside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    monkeypatch.setattr(module, "ROOT", workspace)

    with pytest.raises(RuntimeError, match="inside the workspace"):
        module._workspace_output(outside)


def test_bootstrap_command_builds_into_empty_workspace_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(module, "ROOT", workspace)

    def fake_build(directory: str) -> str:
        output = Path(directory)
        wheel = output / "candidate-bootstrap.whl"
        wheel.write_bytes(b"non-deployable bootstrap wheel")
        return wheel.name

    monkeypatch.setattr(
        module.backend,
        "build_selftest_fixture_bootstrap_wheel",
        fake_build,
    )

    assert module.main(["--wheel-directory", "wheelhouse"]) == 0
    wheel = workspace / "wheelhouse" / "candidate-bootstrap.whl"
    assert wheel.is_file()
    assert capsys.readouterr().out.strip() == str(wheel)


def test_bootstrap_command_rejects_nonempty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    output = workspace / "wheelhouse"
    output.mkdir(parents=True)
    (output / "old.whl").write_bytes(b"old")
    monkeypatch.setattr(module, "ROOT", workspace)

    with pytest.raises(RuntimeError, match="must be empty"):
        module._workspace_output(output)
