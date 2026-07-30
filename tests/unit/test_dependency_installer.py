# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "dependencies" / "install_dependencies.py"


def _module():
    spec = importlib.util.spec_from_file_location("dependency_installer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_inventory_is_exact_and_legacy_is_optional() -> None:
    module = _module()
    payload = module._lock()
    with_legacy = module._sources(payload, with_legacy=True)
    without_legacy = module._sources(payload, with_legacy=False)

    assert {item.key for item in without_legacy} == {
        "symbolica",
        "symbolica-community",
        "gammaloop",
    }
    assert {item.key for item in with_legacy} == {
        *(item.key for item in without_legacy),
        "legacy-amplicol",
    }
    assert all(len(item.revision) == 40 for item in with_legacy)
    legacy = next(item for item in with_legacy if item.key == "legacy-amplicol")
    assert legacy.branch == payload["legacy_amplicol"]["branch"]
    assert legacy.revision == payload["legacy_amplicol"]["revision"]


def test_ufo_loader_uses_the_verified_published_wheel_without_local_patch() -> None:
    module = _module()
    payload = module._lock()
    loader = payload["ufo_model_loader"]
    assert loader["required_version"] == "0.1.7"
    assert loader["latest_verified_published_version"] == "0.1.7"
    assert loader["published_revision"] == ("f3fda32c5e6a673075c345d74a11f12b83c00015")
    assert loader["wheel_sha256"] == (
        "803ae28141ec4be3189cc62469b88da17ca33907791fe99774c2fe756a45edf7"
    )
    assert loader["release_status"] == "verified"
    assert all(patch["target"] != "ufo_model_loader" for patch in payload["patches"])


def test_legacy_oracle_uses_the_pinned_remote_branch_without_local_patches() -> None:
    module = _module()
    payload = module._lock()
    assert all(patch["target"] != "legacy_amplicol" for patch in payload["patches"])
    assert not tuple(
        (module.DEPENDENCIES / "patches" / "legacy-amplicol").glob("*.patch")
    )


def test_venv_reset_bootstraps_with_the_unmoved_base_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    base_python = tmp_path / "base-python"
    active_venv_python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(module.sys, "_base_executable", str(base_python))
    monkeypatch.setattr(module.sys, "executable", str(active_venv_python))

    assert module._venv_bootstrap_python() == base_python


def test_upstream_symjit_revision_archive_and_generic_patch_are_pinned() -> None:
    module = _module()
    payload = module._lock()
    symjit = payload["symjit"]

    assert payload["patches"] == [
        {
            "name": "symjit-raw-plane-descriptor-v1",
            "target": "symjit",
            "path": (
                "patches/symjit/upstream/"
                "0001-Expose-a-stable-raw-P-kernel-plane-descriptor.patch"
            ),
            "sha256": (
                "087071adaaa92a88265d2112ee54546fb0ce8eb979e559997fffaa260e052918"
            ),
            "applies_to_revision": (
                "4e288ce5f3132b05e2a81eb6452c011b9e2bb936"
            ),
        }
    ]
    assert {
        path.relative_to(module.DEPENDENCIES).as_posix()
        for path in (module.DEPENDENCIES / "patches" / "symjit").rglob("*.patch")
    } == {payload["patches"][0]["path"]}
    assert symjit["candidate_version"] == "2.22.0"
    assert symjit["repository"] == "https://github.com/siravan/symjit-crate.git"
    assert symjit["candidate_revision"] == (
        "4e288ce5f3132b05e2a81eb6452c011b9e2bb936"
    )
    assert symjit["archive_sha256"] == (
        "ba279d6135d59570c549b5e826f2ce3bd649b20ebb313c694f81a1fac2ddae5d"
    )
    assert symjit["source_tree_sha256"] == (
        "b309d103b38920c4414ddfcbd7bebaf1d144048c5408b57bfb0ef2086f13df59"
    )
    assert symjit["candidate_tree_sha256"] == (
        "f0cc401d14fd2998845605bb1223f4364e0112e34813080b89a72c02c295be36"
    )
    assert symjit["candidate_tree_sha256"] != symjit["source_tree_sha256"]
    assert hashlib.sha256(b"[]").hexdigest() == module._EMPTY_PATCH_CLOSURE_SHA256
    assert (
        symjit["release_status"]
        == "upstream-git-candidate-with-generic-raw-plane-patch"
    )


def test_contributor_lock_authenticates_the_ordered_generic_patch_closure() -> None:
    module = _module()
    payload = module._lock()

    patches = module._contributor_patches(payload)
    assert [patch.name for patch in patches] == [
        "symjit-raw-plane-descriptor-v1"
    ]
    assert module._patch_state(patches) == payload["patches"]
    assert module._patch_closure_sha256(patches) != hashlib.sha256(b"[]").hexdigest()


def test_generic_contributor_patch_is_authenticated_applied_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    dependencies = tmp_path / "dependencies"
    checkout = dependencies / "checkouts" / "symjit"
    patch_path = dependencies / "patches" / "symjit" / "generic.patch"
    checkout.mkdir(parents=True)
    patch_path.parent.mkdir(parents=True)
    source = checkout / "input.txt"
    source.write_text("before\n", encoding="utf-8")
    patch = (
        "diff --git a/input.txt b/input.txt\n"
        "--- a/input.txt\n"
        "+++ b/input.txt\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n"
    ).encode()
    patch_path.write_bytes(patch)
    revision = "4e288ce5f3132b05e2a81eb6452c011b9e2bb936"
    source_tree = module._source_tree_sha256(checkout)
    source.write_text("after\n", encoding="utf-8")
    candidate_tree = module._source_tree_sha256(checkout)
    source.write_text("before\n", encoding="utf-8")
    payload = {
        "patches": [
            {
                "name": "generic-plane-descriptor",
                "target": "symjit",
                "path": "patches/symjit/generic.patch",
                "sha256": hashlib.sha256(patch).hexdigest(),
                "applies_to_revision": revision,
            }
        ],
        "symjit": {
            "candidate_revision": revision,
            "source_tree_sha256": source_tree,
            "candidate_tree_sha256": candidate_tree,
        },
    }
    monkeypatch.setattr(module, "DEPENDENCIES", dependencies)
    monkeypatch.setattr(module, "CHECKOUTS", dependencies / "checkouts")

    patches = module._contributor_patches(payload)
    assert [item.name for item in patches] == ["generic-plane-descriptor"]
    assert module._patch_closure_sha256(patches) != module._EMPTY_PATCH_CLOSURE_SHA256
    runner = module.Runner(dry_run=False)
    module._apply_contributor_patches(runner, payload)
    module._apply_contributor_patches(runner, payload)

    assert source.read_text(encoding="utf-8") == "after\n"
    assert module._verify_symjit_tree(runner, payload) == candidate_tree


def test_legacy_checkout_clones_the_named_branch_then_pins_its_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "checkouts")
    source = module.Source(
        "legacy-amplicol",
        "https://github.com/rikkert-frederix/AmpliCol.git",
        "60443f327c2203cf92625da2bf0969c27e68a4ac",
        "amplicol_with_patches",
    )
    calls: list[tuple[list[str], Path | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, **_kwargs):
            calls.append(([str(item) for item in command], cwd))
            return subprocess.CompletedProcess(command, 0, "", "")

    module._checkout(FakeRunner(), source, update=False)

    assert calls == [
        (
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--branch",
                "amplicol_with_patches",
                "--single-branch",
                "--no-checkout",
                "https://github.com/rikkert-frederix/AmpliCol.git",
                str(source.path),
            ],
            None,
        ),
        (
            [
                "git",
                "checkout",
                "--detach",
                "60443f327c2203cf92625da2bf0969c27e68a4ac",
            ],
            source.path,
        ),
    ]


def test_contributor_runtime_requirements_use_the_full_hash_locked_closure() -> None:
    module = _module()
    requirements = module._runtime_requirements_text()
    assert "symbolica==" not in requirements
    for requirement in (
        "colorama==0.4.6",
        "numpy==2.4.2",
        "prettytable==3.18.0",
        "progressbar2==4.5.0",
        "python-utils==4.0.0",
        "typing-extensions==4.16.0",
        "ufo-model-loader==0.1.7",
        "wcwidth==0.8.2",
    ):
        assert requirements.count(requirement) == 1
    assert requirements.count("--hash=sha256:") > 20


def test_candidate_dependency_only_build_installs_and_verifies_symbolica(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    venv = tmp_path / ".venv"
    checkouts = tmp_path / "checkouts"
    wheelhouse = tmp_path / "wheelhouse"
    symbolica_wheels = wheelhouse / "symbolica"
    symbolica_wheels.mkdir(parents=True)
    wheel = symbolica_wheels / "symbolica-2.2.0-test.whl"
    wheel.touch()
    calls: list[tuple[list[str], Path | None, dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, cwd, env))
            return subprocess.CompletedProcess(rendered, 0, "", "")

    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "WHEELHOUSE", wheelhouse)

    module._build_candidate_dependency_wheels(
        FakeRunner(),
        {"symbolica": {"candidate_version": "2.2.0"}},
    )

    python = str(venv / "bin" / "python")
    assert calls[0][0][:4] == [python, "-m", "maturin", "build"]
    assert calls[0][1] == checkouts / "symbolica-community"
    assert calls[1][0] == [
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    ]
    assert calls[2][0][:3] == [python, "-I", "-c"]
    assert calls[2][0][-1] == "2.2.0"
    assert "from symbolica import Expression" in calls[2][0][3]
    assert "from symbolica.community.idenso import simplify_color" in calls[2][0][3]
    assert "from symbolica.community.spenso import TensorNetwork" in calls[2][0][3]
    assert calls[2][2]["SYMBOLICA_HIDE_BANNER"] == "1"


@pytest.mark.parametrize("explicit_bootstrap", [None, "1"])
def test_candidate_project_wheel_does_not_force_asset_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    explicit_bootstrap: str | None,
) -> None:
    module = _module()
    venv = tmp_path / ".venv"
    artifacts = tmp_path / "artifacts"
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, env))
            if rendered[1:4] == ["-m", "build", "--wheel"]:
                artifacts.mkdir(parents=True, exist_ok=True)
                (artifacts / "pyamplicol-test.whl").touch()
            return subprocess.CompletedProcess(rendered, 0, "", "")

    if explicit_bootstrap is None:
        monkeypatch.delenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", raising=False)
    else:
        monkeypatch.setenv(
            "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP",
            explicit_bootstrap,
        )
    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "ARTIFACTS", artifacts)

    module._build_candidate_project_wheel(FakeRunner())

    assert len(calls) == 2
    build_command, build_environment = calls[0]
    assert build_command[1:4] == ["-m", "build", "--wheel"]
    assert build_environment is not None
    assert build_environment["PYAMPLICOL_BUILD_MODE"] == "candidate"
    install_command, install_environment = calls[1]
    assert install_command[1:4] == ["-m", "pip", "install"]
    assert install_environment is not None
    if explicit_bootstrap is None:
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in build_environment
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in install_environment
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in os.environ
    else:
        assert (
            build_environment["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"]
            == explicit_bootstrap
        )
        assert (
            install_environment["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"]
            == explicit_bootstrap
        )
        assert os.environ["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"] == explicit_bootstrap


def test_dependency_only_and_no_build_are_mutually_exclusive() -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--dependencies-only", "--no-build"])


def test_dependency_only_cli_skips_the_project_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    payload: dict[str, object] = {"patches": []}
    monkeypatch.setattr(module, "_lock", lambda: payload)
    monkeypatch.setattr(module, "_sources", lambda *_args, **_kwargs: ())
    for name in (
        "_ensure_just",
        "_ensure_venv",
        "_materialize_symjit",
        "_configure_sources",
        "_verify_symjit_tree",
        "_write_cargo_config",
        "_write_candidate_lock",
        "_write_state",
    ):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_build_candidate_dependency_wheels",
        lambda *_args: calls.append("dependencies"),
    )
    monkeypatch.setattr(
        module,
        "_build_candidate_wheels",
        lambda *_args: calls.append("dependencies-and-project"),
    )

    assert module.main(["--dry-run", "--dependencies-only"]) == 0
    assert calls == ["dependencies"]


def test_toml_section_replacement_is_idempotent() -> None:
    module = _module()
    original = '[package]\nname = "x"\n\n[dependencies]\na = "1"\n'
    once = module._replace_section(original, "dependencies", 'b = "2"')
    twice = module._replace_section(once, "dependencies", 'b = "2"')
    assert once == twice
    assert 'a = "1"' not in once
    assert once.count('b = "2"') == 1


def test_symjit_rlib_manifest_validation_does_not_rewrite_source(
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = tmp_path / "Cargo.toml"
    pristine = (
        b'[package]\nname = "symjit"\nversion = "2.22.0"\n\n'
        b'[lib]\ncrate-type = ["rlib"]\n'
    )
    manifest.write_bytes(pristine)

    module._require_symjit_rlib_manifest(manifest, expected_version="2.22.0")

    assert manifest.read_bytes() == pristine

    manifest.write_text(
        '[package]\nname = "symjit"\nversion = "2.22.0"\n\n'
        '[lib]\ncrate-type = ["rlib", "cdylib"]\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SetupError, match="rlib-only"):
        module._require_symjit_rlib_manifest(manifest)

    manifest.write_text(
        '[package]\nname = "symjit"\nversion = "2.21.0"\n\n'
        '[lib]\ncrate-type = ["rlib"]\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SetupError, match="wrong package version"):
        module._require_symjit_rlib_manifest(
            manifest,
            expected_version="2.22.0",
        )


def test_symjit_archive_materialization_preserves_the_pristine_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    prefix = "symjit-test-revision"
    files = {
        "Cargo.toml": (
            b'[package]\nname = "symjit"\nversion = "2.22.0"\n\n'
            b'[lib]\ncrate-type = ["rlib"]\n'
        ),
        "rust/direct.rs": b"pub fn direct() {}\n",
    }
    archive_stream = io.BytesIO()
    with tarfile.open(fileobj=archive_stream, mode="w:gz") as archive:
        for relative, content in files.items():
            entry = tarfile.TarInfo(f"{prefix}/{relative}")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    archive_bytes = archive_stream.getvalue()

    expected = tmp_path / "expected"
    for relative, content in files.items():
        path = expected / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    tree_sha256 = module._source_tree_sha256(expected)
    payload = {
        "symjit": {
            "candidate_version": "2.22.0",
            "source_url": "https://example.invalid/symjit.tar.gz",
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "archive_prefix": prefix,
            "source_tree_sha256": tree_sha256,
            "candidate_tree_sha256": tree_sha256,
        }
    }
    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "checkouts")
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda _url: io.BytesIO(archive_bytes),
    )

    module._materialize_symjit(module.Runner(dry_run=False), payload)

    installed = module.CHECKOUTS / "symjit"
    assert module._source_tree_sha256(installed) == tree_sha256
    assert {
        path.relative_to(installed).as_posix(): path.read_bytes()
        for path in installed.rglob("*")
        if path.is_file()
    } == files


def test_candidate_community_lock_is_resolved_from_the_upstream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    community = tmp_path / "checkouts" / "symbolica-community"
    community.mkdir(parents=True)
    lock = community / "Cargo.lock"
    lock.write_text("stale generated lock\n", encoding="utf-8")
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append(rendered)
            assert cwd == community
            assert capture is True
            if rendered[:2] == ["git", "show"]:
                return subprocess.CompletedProcess(command, 0, "upstream lock\n", "")
            if "--locked" not in rendered:
                assert lock.read_text(encoding="utf-8") == "upstream lock\n"
                lock.write_text("path-resolved lock\n", encoding="utf-8")
            else:
                assert lock.read_text(encoding="utf-8") == "path-resolved lock\n"
            return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "checkouts")
    monkeypatch.setattr(module, "_configure_source_manifests", lambda _runner: None)

    module._configure_sources(FakeRunner())

    assert calls == [
        ["git", "show", "HEAD:Cargo.lock"],
        ["cargo", "metadata", "--format-version", "1"],
        ["cargo", "metadata", "--locked", "--format-version", "1"],
    ]
    assert lock.read_text(encoding="utf-8") == "path-resolved lock\n"


def test_installer_tree_fingerprint_matches_content_and_ignores_build_cache(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "input.txt"
    payload.write_text("first\n", encoding="utf-8")
    first = module._source_tree_sha256(source)
    payload.write_text("second\n", encoding="utf-8")
    second = module._source_tree_sha256(source)
    assert first != second
    target = source / "target"
    target.mkdir()
    (target / "output").write_text("build\n", encoding="utf-8")
    assert module._source_tree_sha256(source) == second


def test_canonical_release_lock_has_no_candidate_path_packages() -> None:
    module = _module()
    module._validate_release_cargo_lock(ROOT / "Cargo.lock")


def test_candidate_lock_is_seeded_without_mutating_canonical_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    release_lock = project / "Cargo.lock"
    release_lock.write_bytes(b"canonical release lock\n")
    (project / "rust").mkdir()
    cargo_config = project / ".cargo" / "config.toml"
    cargo_config.parent.mkdir()
    cargo_config.write_text("[patch.crates-io]\n", encoding="utf-8")
    candidate_lock = project / "dependencies" / "candidate-Cargo.lock"
    candidate_lock.parent.mkdir()
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            assert cwd is not None and capture is True
            calls.append(list(command))
            staged_lock = Path(cwd) / "Cargo.lock"
            if len(calls) == 1:
                assert staged_lock.read_bytes() == b"canonical release lock\n"
            elif len(calls) == 2:
                assert staged_lock.read_bytes() == b"canonical release lock\n"
                staged_lock.write_bytes(b"candidate path lock\n")
            else:
                assert staged_lock.read_bytes() == b"candidate path lock\n"
            return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setattr(module, "CARGO_CONFIG", cargo_config)
    monkeypatch.setattr(module, "CANDIDATE_LOCK", candidate_lock)
    monkeypatch.setattr(module, "_validate_release_cargo_lock", lambda _path: None)
    monkeypatch.setattr(module, "_validate_candidate_cargo_lock", lambda _path: None)
    projected: list[Path] = []
    monkeypatch.setattr(
        module,
        "_rewrite_candidate_requirements",
        lambda root: projected.append(root),
    )

    module._write_candidate_lock(FakeRunner())

    assert calls == [
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        ["cargo", "metadata", "--format-version", "1"],
        ["cargo", "metadata", "--locked", "--format-version", "1"],
    ]
    assert release_lock.read_bytes() == b"canonical release lock\n"
    assert candidate_lock.read_bytes() == b"candidate path lock\n"
    assert len(projected) == 1
    assert projected[0].parent != project


def test_candidate_dependency_projection_rewrites_only_the_isolated_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = tmp_path / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        'symbolica = { version = "=2.1.0", default-features = false }\n'
        'symjit = { version = "=2.22.0", default-features = false }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_lock",
        lambda: {
            "symbolica": {
                "rust_version": "2.1.0",
                "candidate_version": "2.2.0",
            },
            "symjit": {"candidate_version": "2.22.1"},
        },
    )
    monkeypatch.setattr(
        module,
        "_release_lock",
        lambda: {"symjit": {"version": "2.22.0"}},
    )

    module._rewrite_candidate_requirements(tmp_path)

    projected = manifest.read_text(encoding="utf-8")
    assert 'symbolica = { version = "=2.2.0"' in projected
    assert 'symjit = { version = "=2.22.1"' in projected
