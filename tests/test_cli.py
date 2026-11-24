from pathlib import Path

import pytest

from sparjax.cli import main


def test_cfr_solve_and_br(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "kuhn.npz"
    main(["cfr-solve", "--game", "kuhn_poker", "--iters", "200", "--out", str(out)])
    assert out.exists()
    text = capsys.readouterr().out
    assert "saved 12 information states" in text
    assert "exploitability" in text

    main(["cfr-br", "--policy", str(out)])
    assert "best-response value" in capsys.readouterr().out


def test_cfr_solve_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    main(["cfr-solve", "--game", "kuhn_poker", "--iters", "3", "--no-report"])
    assert (tmp_path / "policies" / "kuhn_poker_cfr_plus_3.npz").exists()


def test_cfr_solve_rejects_unknown_algorithm() -> None:
    with pytest.raises(SystemExit):
        main(["cfr-solve", "--game", "kuhn_poker", "--algorithm", "nope"])
