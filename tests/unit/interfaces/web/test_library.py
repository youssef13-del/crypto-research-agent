import builtins
import importlib
import sys
from datetime import UTC, datetime, timedelta

import pytest

from crypto_research.domain.history import ResearchRunSummary
from crypto_research.interfaces.web.pages.library import (
    _library_frame,
    bulk_delete_confirmation,
    selected_run_summaries,
)


def _summary(index: int, *, pinned: bool = False) -> ResearchRunSummary:
    timestamp = datetime(2026, 8, 16, 12, tzinfo=UTC) - timedelta(hours=index)
    return ResearchRunSummary(
        id=f"00000000-0000-0000-0000-{index:012d}",
        created_at=timestamp,
        completed_at=timestamp,
        state="complete",
        question=f"Research report {index}",
        assets=("BTC/USD",) if index == 0 else ("ETH/USD",),
        capabilities=("market", "risk"),
        exchange="kraken",
        timeframe="1h",
        pinned=pinned,
        evidence_count=index + 2,
    )


def test_library_frame_contains_readable_management_columns() -> None:
    frame = _library_frame((_summary(0, pinned=True), _summary(1)))

    assert list(frame.columns) == [
        "Pinned",
        "Assets",
        "Topics",
        "Status",
        "Exchange",
        "Timeframe",
        "Evidence",
        "Completed",
    ]
    assert frame.iloc[0].to_dict()["Pinned"] is True
    assert frame.iloc[0].to_dict()["Topics"] == "Market, Risk"


def test_selected_rows_map_to_original_summaries_and_ignore_invalid_duplicates() -> None:
    summaries = (_summary(0), _summary(1), _summary(2))

    selected = selected_run_summaries(summaries, (2, 2, -1, 99, 0))

    assert selected == (summaries[2], summaries[0])


def test_bulk_delete_confirmation_is_exact_and_bounded() -> None:
    assert bulk_delete_confirmation(2) == "DELETE 2 REPORTS"
    assert bulk_delete_confirmation(100) == "DELETE 100 REPORTS"
    with pytest.raises(ValueError, match="between 2 and 100"):
        bulk_delete_confirmation(1)
    with pytest.raises(ValueError, match="between 2 and 100"):
        bulk_delete_confirmation(101)


def test_app_startup_does_not_import_pdf_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def import_without_pdf_renderer(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "crypto_research.interfaces.web.pdf_report":
            raise ModuleNotFoundError("PDF dependencies are not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_pdf_renderer)
    sys.modules.pop("crypto_research.interfaces.web.app", None)
    sys.modules.pop("crypto_research.interfaces.web.pages.library", None)

    module = importlib.import_module("crypto_research.interfaces.web.app")

    assert callable(module.main)
