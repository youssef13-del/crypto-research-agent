import sys
from pathlib import Path

import pytest
from streamlit.web import cli as streamlit_cli

from crypto_research.interfaces.web import launcher


def test_launcher_forwards_streamlit_options_before_script_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_main(*, prog_name: str) -> int:
        captured["prog_name"] = prog_name
        captured["argv"] = list(sys.argv)
        return 0

    monkeypatch.setattr(streamlit_cli, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["chainscope", "--server.headless=true", "--server.port=8765"],
    )

    with pytest.raises(SystemExit, match="0"):
        launcher.main()

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[:2] == ["streamlit", "run"]
    assert Path(str(argv[2])).name == "streamlit_app.py"
    assert Path(str(argv[2])).parent.name == "web"
    assert argv[3:] == ["--server.headless=true", "--server.port=8765"]
    assert "--" not in argv
    assert captured["prog_name"] == "chainscope"
