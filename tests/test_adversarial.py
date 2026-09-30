from scripts.run_adversarial_tests import main


def test_adversarial_suite_passes_offline(capsys, tmp_path):
    assert main(["--offline", "--log-dir", str(tmp_path)]) == 0
    assert "Guard mode: offline" in capsys.readouterr().out
    assert (tmp_path / "events.jsonl").exists()
