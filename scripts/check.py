"""Run the repository's required local and CI quality checks."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PYTHON = (3, 14)

_PACKAGE_SMOKE = """
from importlib import import_module
from importlib.metadata import distribution

package = distribution("crypto-research-agent")
expected_entry_points = {
    "chainscope": "crypto_research.interfaces.web.launcher:main",
    "crypto-research": "crypto_research.interfaces.cli:main",
}
actual_entry_points = {
    entry_point.name: entry_point.value
    for entry_point in package.entry_points
}
for name, target in expected_entry_points.items():
    actual = actual_entry_points.get(name)
    if actual != target:
        raise SystemExit(
            f"entry point {name!r} should target {target!r}, found {actual!r}"
        )

for module_name in (
    "crypto_research",
    "crypto_research.bootstrap",
    "crypto_research.interfaces.cli",
    "crypto_research.interfaces.web",
    "crypto_research.orchestration.runtime",
    "crypto_research.agents.registry",
    "crypto_research.agents.onchain",
    "crypto_research.tools.onchain",
    "crypto_research.storage",
    "crypto_research.orchestration.planning",
    "crypto_research.domain.research",
    "crypto_research.domain.market",
    "crypto_research.domain.forecast",
    "crypto_research.forecasting",
    "crypto_research.interfaces.web.pages.home",
    "crypto_research.interfaces.web.pages.research",
    "crypto_research.interfaces.web.pages.dashboard",
    "crypto_research.interfaces.web.pages.library",
):
    import_module(module_name)

print(f"package/import smoke passed for crypto-research-agent {package.version}")
"""

_ENCODING_SMOKE = r"""
from pathlib import Path

root = Path.cwd()
paths = [root / "README.md", *(root / "src").rglob("*.py")]
markers = ("\ufffd", "\u00c2", "\u00c3", "\u00e2\u20ac")
for path in paths:
    text = path.read_text(encoding="utf-8")
    if any(marker in text for marker in markers):
        raise SystemExit(f"possible mojibake in {path.relative_to(root)}")
print("UTF-8 source hygiene passed")
"""


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    command: tuple[str, ...]


def _checks() -> tuple[Check, ...]:
    python = sys.executable
    python_sources = ("src", "tests", "scripts")
    pytest_command = (python, "-m", "pytest", "-q")
    if os.name == "nt":
        basetemp = tempfile.mkdtemp(prefix="crypto-research-pytest-")
        pytest_command = (*pytest_command, "--basetemp", basetemp)
    return (
        Check("Ruff lint", (python, "-m", "ruff", "check", *python_sources)),
        Check(
            "Ruff format",
            (python, "-m", "ruff", "format", "--check", *python_sources),
        ),
        Check("UTF-8 source hygiene", (python, "-c", _ENCODING_SMOKE)),
        Check("mypy", (python, "-m", "mypy", "src", "tests")),
        Check("pytest", pytest_command),
        Check("dependency consistency", (python, "-m", "pip", "check")),
        Check("package/import smoke", (python, "-c", _PACKAGE_SMOKE)),
    )


def main() -> int:
    if sys.version_info[:2] != REQUIRED_PYTHON:
        required = ".".join(str(part) for part in REQUIRED_PYTHON)
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(
            f"error: ChainScope checks require Python {required}; "
            f"this interpreter is Python {current}.",
            file=sys.stderr,
        )
        return 2

    for check in _checks():
        print(f"\n==> {check.name}", flush=True)
        completed = subprocess.run(check.command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode:
            print(
                f"\n{check.name} failed with exit code {completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
