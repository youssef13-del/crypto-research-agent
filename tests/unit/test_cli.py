import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crypto_research.domain.research import AnalysisRequest, ResearchReport
from crypto_research.interfaces.cli import _console_print, build_parser, main
from crypto_research.orchestration.events import WorkflowNode
from crypto_research.orchestration.runtime import ResearchOutcome


def test_cli_accepts_analyze_and_discover_subcommands() -> None:
    analyze = build_parser().parse_args(
        [
            "analyze",
            "--assets",
            "BTC",
            "ETH",
            "--topics",
            "market",
            "news",
            "--exchange",
            "coinbase",
        ]
    )
    discover = build_parser().parse_args(["discover", "--timeframe", "4h"])

    assert analyze.assets == ["BTC", "ETH"]
    assert analyze.topics == ["market", "news"]
    assert analyze.exchange == "coinbase"
    assert discover.command == "discover"
    assert discover.timeframe == "4h"


def test_cli_rejects_removed_chat_subcommand() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["chat", "Why does RSI matter?"])

    assert error.value.code == 2


def test_cli_runs_research_and_writes_public_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    runtime = MagicMock()
    runtime.ask.return_value = _research_outcome()
    monkeypatch.setattr(
        "crypto_research.interfaces.cli.create_research_runtime",
        lambda _, *, owner_id: runtime,
    )
    output = tmp_path / "research.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto-research",
            "--output",
            str(output),
            "analyze",
            "--assets",
            "BTC",
            "ETH",
            "--topics",
            "market",
            "news",
        ],
    )

    assert main() == 0
    captured = capsys.readouterr()

    assert not captured.err
    assert "Analysis completed" in captured.out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["request"]["comparison_symbols"] == ["BTC/USD", "ETH/USD"]
    action = runtime.ask.call_args.args[0]
    assert action.request.user_intent == "Compare BTC, ETH: Market behavior + Recent news"


def test_cli_hides_unexpected_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "GROQ_API_KEY=must-not-leak"
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    monkeypatch.setattr(
        sys,
        "argv",
        ["crypto-research", "analyze", "--assets", "BTC", "--topics", "market"],
    )
    runtime = MagicMock()
    runtime.ask.side_effect = RuntimeError(secret)
    monkeypatch.setattr(
        "crypto_research.interfaces.cli.create_research_runtime",
        lambda _, *, owner_id: runtime,
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 1
    assert secret not in capsys.readouterr().err


def test_console_output_degrades_safely_on_legacy_windows_encoding() -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")

    _console_print("non-breaking text", stream=stream)
    stream.flush()

    assert buffer.getvalue().decode("cp1252").splitlines() == ["non-breaking text"]
    stream.detach()


def _research_outcome() -> ResearchOutcome:
    request = AnalysisRequest(
        user_intent="Compare BTC and ETH",
        comparison_symbols=["BTC/USD", "ETH/USD"],
    )
    return ResearchOutcome(
        research_report=ResearchReport(request=request, status="complete"),
        agents=(WorkflowNode.MARKET_AGENT, WorkflowNode.NEWS_AGENT),
        route=(WorkflowNode.MARKET_AGENT, WorkflowNode.NEWS_AGENT),
        requested_capabilities=(),
        warnings=(),
        errors=(),
    )
