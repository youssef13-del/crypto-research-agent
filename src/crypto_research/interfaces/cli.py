"""Thin command-line client for ChainScope research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from crypto_research.bootstrap import create_research_runtime
from crypto_research.config import Settings
from crypto_research.domain.core import ResearchCapability
from crypto_research.domain.forecast import ForecastSettings
from crypto_research.orchestration.planning import compile_guided_research_plan
from crypto_research.orchestration.runtime import ResearchOutcome
from crypto_research.shared.security import redact_secrets

_TOPIC_TO_CAPABILITY = {capability.value: capability for capability in ResearchCapability}
_ANALYSIS_TOPICS = (
    "market",
    "risk",
    "derivatives",
    "fundamentals",
    "defi",
    "onchain",
    "news",
    "forecast",
)


def main() -> int:
    _configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args()
    try:
        settings = Settings.load_application()
        runtime = create_research_runtime(settings, owner_id="cli")
        if args.command == "discover":
            plan = compile_guided_research_plan(
                [],
                [ResearchCapability.DISCOVERY],
                mode="discovery",
                exchange=args.exchange,
                timeframe=args.timeframe,
            )
            result = runtime.ask(plan.action)
        elif args.command == "analyze":
            caps = [_TOPIC_TO_CAPABILITY[item] for item in args.topics]
            forecast_settings = (
                ForecastSettings(
                    model_id=args.forecast_model,
                    horizon_hours=args.forecast_horizon,
                    confidence_level=args.forecast_confidence,
                    lookback_candles=args.forecast_lookback,
                )
                if ResearchCapability.FORECAST in caps
                else None
            )
            plan = compile_guided_research_plan(
                args.assets,
                caps,
                mode="asset",
                exchange=args.exchange,
                timeframe=args.timeframe,
                forecast_settings=forecast_settings,
            )
            result = runtime.ask(plan.action)
        else:  # pragma: no cover - argparse guards this
            parser.error("Choose analyze or discover.")
        public_payload = result.public_payload()
    except ValidationError as exc:
        parser.exit(2, f"error: {_validation_error_message(exc)}\n")
    except ValueError as exc:
        parser.exit(2, f"error: {redact_secrets(str(exc))}\n")
    except Exception:
        parser.exit(1, "error: the research request could not be completed.\n")

    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(public_payload, indent=2), encoding="utf-8")
    except OSError, TypeError:
        parser.exit(1, "error: the result could not be written to the requested output path.\n")
    print_summary(result, output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crypto-research")
    parser.add_argument("--output", default="outputs/research.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run selected research topics.")
    analyze.add_argument(
        "--assets",
        nargs="+",
        required=True,
        help="One to four assets, e.g. BTC ETH.",
    )
    analyze.add_argument(
        "--topics",
        nargs="+",
        required=True,
        choices=_ANALYSIS_TOPICS,
        help="One or more explicit research topics.",
    )
    analyze.add_argument(
        "--forecast-model",
        default="gradient_boosting_huber",
        choices=("gradient_boosting_huber", "ridge"),
    )
    analyze.add_argument(
        "--forecast-horizon",
        type=int,
        default=24,
        choices=(4, 8, 12, 24, 48),
    )
    analyze.add_argument(
        "--forecast-confidence",
        type=float,
        default=0.8,
        choices=(0.8, 0.9),
    )
    analyze.add_argument(
        "--forecast-lookback",
        type=int,
        default=750,
        choices=range(400, 2001),
        metavar="400-2000",
    )
    _add_market_options(analyze)

    discover = subparsers.add_parser("discover", help="Run Market & Risk discovery.")
    _add_market_options(discover)

    return parser


def _add_market_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exchange",
        default="kraken",
        choices=["kraken", "coinbase", "binance"],
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    )


def _validation_error_message(error: ValidationError) -> str:
    first_error = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    label = location or "configuration"
    return f"invalid {label}: {first_error.get('msg', 'validation failed')}"


def print_summary(result: ResearchOutcome, output_path: Path) -> None:
    report = result.research_report
    _console_print("Analysis completed")
    _console_print(f"agents: {', '.join(node.value for node in result.agents)}")
    _console_print(f"saved: {output_path}")
    for answer in report.agent_answers:
        _console_print(f"\n{answer.agent.replace('_', ' ').title()}: {answer.answer}")
        for limitation in answer.limitations:
            _console_print(f"limitation: {limitation}")
    _console_print("Educational research only. Not financial advice.")


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").replace("-", "").casefold()
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding == "utf8" or not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except OSError, ValueError:
            continue


def _console_print(value: object, *, stream: TextIO | None = None) -> None:
    target = stream or sys.stdout
    try:
        print(str(value), file=target)
    except UnicodeEncodeError:
        encoding = target.encoding or "ascii"
        safe_text = str(value).encode(encoding, errors="replace").decode(encoding)
        print(safe_text, file=target)


if __name__ == "__main__":
    raise SystemExit(main())
