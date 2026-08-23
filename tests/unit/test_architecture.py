import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "crypto_research"

ALLOWED = {
    "__init__": set(),
    "agents": {"agents", "domain", "forecasting", "llm", "orchestration", "shared", "tools"},
    "bootstrap": {
        "agents",
        "config",
        "domain",
        "forecasting",
        "llm",
        "orchestration",
        "storage",
        "tools",
    },
    "config": {"config"},
    "domain": {"domain", "shared"},
    "interfaces": {
        "bootstrap",
        "config",
        "domain",
        "interfaces",
        "orchestration",
        "shared",
        "storage",
        "tools",
    },
    "llm": {"config", "domain", "llm", "shared"},
    "orchestration": {
        "agents",
        "domain",
        "forecasting",
        "llm",
        "orchestration",
        "shared",
        "storage",
        "tools",
    },
    "shared": {"shared"},
    "storage": {"domain", "shared", "storage"},
    "tools": {"domain", "shared", "tools"},
}


def test_layer_dependency_budgets() -> None:
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        owner = _owner(path)
        if owner not in ALLOWED:
            continue
        for imported in _imports(path):
            target = imported.split(".", maxsplit=1)[0] if imported else "(root)"
            if target not in ALLOWED[owner]:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} imports crypto_research.{imported}"
                )
    assert violations == [], "Undeclared package dependencies:\n" + "\n".join(violations)


def test_tools_do_not_depend_on_higher_layers() -> None:
    forbidden = {"agents", "bootstrap", "interfaces", "llm", "orchestration"}
    violations = _forbidden_imports(PACKAGE_ROOT / "tools", forbidden)
    assert violations == [], "Tool dependency violations:\n" + "\n".join(violations)


def test_agent_folders_do_not_depend_on_peers_or_orchestration() -> None:
    violations: list[str] = []
    root = PACKAGE_ROOT / "agents"
    agent_folders = {"market", "news", "fundamentals", "onchain", "forecast"}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue
        owner = relative.parts[0]
        violations.extend(
            f"{relative} imports crypto_research.{imported}"
            for imported in _imports(path)
            if imported.startswith("orchestration")
            or (
                imported.startswith("agents.")
                and (target := imported.split(".", maxsplit=2)[1]) in agent_folders
                and target != owner
            )
        )
    assert violations == [], "Agent dependency violations:\n" + "\n".join(violations)


def test_public_domain_exports_are_explicit() -> None:
    tree = ast.parse((PACKAGE_ROOT / "domain" / "__init__.py").read_text(encoding="utf-8"))
    assert not [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and any(name.name == "*" for name in node.names)
    ]


def test_disabled_cli_import_does_not_load_optional_provider_stacks() -> None:
    _assert_fresh_import_excludes(
        "crypto_research.interfaces.cli",
        ("ccxt", "feedparser", "langchain_openai", "numpy", "pandas", "sklearn"),
    )


def test_evidence_import_does_not_load_market_provider_stacks() -> None:
    _assert_fresh_import_excludes(
        "crypto_research.orchestration.evidence", ("ccxt", "numpy", "pandas")
    )


def test_relative_imports_are_included_in_the_dependency_audit(tmp_path: Path) -> None:
    package = tmp_path / "feature"
    package.mkdir()
    module = package / "module.py"
    module.write_text("from ..tools import cache\n", encoding="utf-8")
    assert _imports(module, root=tmp_path) == ["tools"]


def _forbidden_imports(root: Path, forbidden: set[str]) -> list[str]:
    violations: list[str] = []
    for path in root.rglob("*.py"):
        for imported in _imports(path):
            target = imported.split(".", maxsplit=1)[0] if imported else "(root)"
            if target in forbidden:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} imports crypto_research.{imported}"
                )
    return violations


def _assert_fresh_import_excludes(module: str, excluded: tuple[str, ...]) -> None:
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"unexpected = sorted(set({excluded!r}) & set(sys.modules))\n"
        "raise SystemExit(f'loaded optional modules: {unexpected}' if unexpected else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=PACKAGE_ROOT.parents[1], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def _owner(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else path.stem


def _imports(path: Path, *, root: Path = PACKAGE_ROOT) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(value for name in node.names if (value := _absolute_import(name.name)))
        elif isinstance(node, ast.ImportFrom):
            value = _from_import(path, node, root=root)
            if value is not None:
                imports.append(value)
    return imports


def _absolute_import(name: str) -> str | None:
    if name == "crypto_research":
        return ""
    return name.removeprefix("crypto_research.") if name.startswith("crypto_research.") else None


def _from_import(path: Path, node: ast.ImportFrom, *, root: Path) -> str | None:
    if node.level == 0:
        return _absolute_import(node.module or "")
    parts = list(path.relative_to(root).parent.parts)
    hops = node.level - 1
    if hops > len(parts):
        return None
    base = parts[: len(parts) - hops]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)
