"""Offline tests for common/kaggle_cli.py — all subprocess calls are faked.

The real CLI hits the network; here we monkeypatch ``subprocess.run`` and the
credential path constants so the wrapper logic (argument building, error
propagation, create-vs-version flow) is exercised without any API calls.
"""

import subprocess
import sys

import pytest


@pytest.fixture
def cli_mod(monkeypatch, tmp_path):
    import common.kaggle_cli as k

    cred_dir = tmp_path / ".kaggle"
    cred_dir.mkdir()
    monkeypatch.setattr(k, "KAGGLE_CREDENTIAL_DIR", cred_dir)
    monkeypatch.setattr(k, "KAGGLE_JSON", cred_dir / "kaggle.json")
    monkeypatch.setattr(k, "CREDENTIALS_JSON", cred_dir / "credentials.json")
    return k, cred_dir


def _fake_run(monkeypatch, returncode=0, stdout="", stderr=""):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args, returncode=returncode, stdout=stdout, stderr=stderr
        )

    monkeypatch.setattr("common.kaggle_cli.subprocess.run", fake_run)
    return calls


def test_cli_available_uses_python_m(monkeypatch):
    import common.kaggle_cli as k

    calls = _fake_run(monkeypatch, returncode=0)
    assert k.cli_available() is True
    assert calls[0] == [sys.executable, "-m", "kaggle", "--version"]


def test_cli_available_false_on_failure(monkeypatch):
    import common.kaggle_cli as k

    _fake_run(monkeypatch, returncode=1)
    assert k.cli_available() is False


def test_cli_available_false_on_oserror(monkeypatch):
    import common.kaggle_cli as k

    def boom(*args, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr("common.kaggle_cli.subprocess.run", boom)
    assert k.cli_available() is False


def test_has_credentials_detects_api_key(cli_mod):
    k, cred_dir = cli_mod
    assert k.has_credentials() is False
    (cred_dir / "kaggle.json").write_text("{}", encoding="utf-8")
    assert k.has_credentials() is True


def test_has_credentials_detects_oauth(cli_mod):
    k, cred_dir = cli_mod
    (cred_dir / "credentials.json").write_text("{}", encoding="utf-8")
    assert k.has_credentials() is True


def test_verify_auth_runs_readonly_probe(monkeypatch, cli_mod):
    k, _ = cli_mod
    k.cli_available = lambda: True
    (k.KAGGLE_JSON).write_text("{}", encoding="utf-8")
    calls = _fake_run(monkeypatch, returncode=0)
    assert k.verify_auth() is True
    assert "datasets" in calls[0] and "list" in calls[0]


def test_verify_auth_false_without_credentials(cli_mod):
    k, _ = cli_mod
    k.cli_available = lambda: True
    assert k.verify_auth() is False  # no credentials -> no probe even attempted


def test_auth_status_strings(monkeypatch, cli_mod):
    k, cred_dir = cli_mod
    k.cli_available = lambda: False
    assert k.auth_status() == "cli_missing"
    k.cli_available = lambda: True
    assert k.auth_status() == "no_credentials"
    (cred_dir / "kaggle.json").write_text("{}", encoding="utf-8")
    k.verify_auth = lambda: False
    assert k.auth_status() == "unauthenticated"
    k.verify_auth = lambda: True
    assert k.auth_status() == "authenticated"


def test_datasets_download_builds_args(monkeypatch, cli_mod, tmp_path):
    k, _ = cli_mod
    calls = _fake_run(monkeypatch, returncode=0)
    target = k.datasets_download("owner/corpus", str(tmp_path))
    assert target == tmp_path
    assert "datasets" in calls[0]
    assert "download" in calls[0]
    assert "owner/corpus" in calls[0]
    assert str(tmp_path) in calls[0]
    assert "--unzip" in calls[0]
    assert "-q" in calls[0]


def test_datasets_download_raises_on_failure(monkeypatch, cli_mod, tmp_path):
    k, _ = cli_mod
    _fake_run(monkeypatch, returncode=1, stderr="401 Unauthorized")
    with pytest.raises(k.KaggleCLIError, match="401 Unauthorized"):
        k.datasets_download("owner/corpus", str(tmp_path))


def test_write_datasets_metadata_id_and_no_id(tmp_path):
    from common.kaggle_cli import write_datasets_metadata

    with_id = write_datasets_metadata(
        str(tmp_path / "v"), "Title", owner_slug="me", dataset_slug="ckpts"
    )
    meta = __import__("json").loads(with_id.read_text(encoding="utf-8"))
    assert meta["id"] == "me/ckpts"

    no_id = write_datasets_metadata(str(tmp_path / "n"), "Title")
    meta = __import__("json").loads(no_id.read_text(encoding="utf-8"))
    assert "id" not in meta  # create-mode payloads must omit id


def test_datasets_version_passes_message(monkeypatch, cli_mod, tmp_path):
    k, _ = cli_mod
    calls = _fake_run(monkeypatch, returncode=0)
    k.datasets_version(str(tmp_path), message="v2 snapshot")
    assert "version" in calls[0]
    assert "-m" in calls[0] and "v2 snapshot" in calls[0]


def test_ensure_dataset_creates_then_versions(monkeypatch, cli_mod, tmp_path):
    """First push: version fails with 'not found' -> create -> version succeeds."""
    k, _ = cli_mod
    calls = []
    outcomes = iter([1, 0, 0])  # version fails, create ok, version ok

    def fake_run(args, **kwargs):
        calls.append(args)
        rc = next(outcomes)
        stderr = "Dataset not found" if rc != 0 else ""
        return subprocess.CompletedProcess(args, rc, stdout="", stderr=stderr)

    monkeypatch.setattr("common.kaggle_cli.subprocess.run", fake_run)
    action = k.ensure_dataset(
        str(tmp_path), "Checkpoints", "me", "ckpts", quiet=True
    )
    assert action == "created"
    cmds = [a[4] for a in calls]  # [py, -m, kaggle, datasets, <subcommand>, ...]
    assert cmds == ["version", "create", "version"]


def test_ensure_dataset_versions_directly(monkeypatch, cli_mod, tmp_path):
    k, _ = cli_mod
    _fake_run(monkeypatch, returncode=0)
    action = k.ensure_dataset(str(tmp_path), "Checkpoints", "me", "ckpts")
    assert action == "versioned"


def test_ensure_dataset_rethrows_non_notfound(monkeypatch, cli_mod, tmp_path):
    k, _ = cli_mod
    _fake_run(monkeypatch, returncode=1, stderr="413 Payload too large")
    with pytest.raises(k.KaggleCLIError, match="413"):
        k.ensure_dataset(str(tmp_path), "Checkpoints", "me", "ckpts")


def test_main_auth(monkeypatch, cli_mod, capsys):
    k, _ = cli_mod
    k.auth_status = lambda: "authenticated"
    assert k.main(["auth"]) == 0
    k.auth_status = lambda: "no_credentials"
    assert k.main(["auth"]) == 1
    out = capsys.readouterr().out
    assert "auth status" in out
