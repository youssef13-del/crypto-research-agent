import tomllib
from pathlib import Path

from crypto_research.config import Settings

PROJECT_ROOT = Path(__file__).parents[3]
EXPECTED_KEYS = {name.upper() for name in Settings.model_fields}


def test_environment_example_matches_the_settings_contract() -> None:
    values = _dotenv_values(PROJECT_ROOT / ".env.example")

    assert set(values) == EXPECTED_KEYS
    Settings.from_mapping(values)


def test_streamlit_example_matches_the_settings_contract() -> None:
    path = PROJECT_ROOT / ".streamlit" / "secrets.toml.example"
    values = tomllib.loads(path.read_text(encoding="utf-8"))
    auth = values.pop("auth")

    assert set(values) == EXPECTED_KEYS
    Settings.from_mapping(values)
    assert set(auth) == {"redirect_uri", "cookie_secret", "auth0"}
    assert set(auth["auth0"]) == {
        "client_id",
        "client_secret",
        "server_metadata_url",
    }


def test_auth0_setup_documents_the_required_logout_allowlist() -> None:
    callback = "http://localhost:8501/oauth2callback"
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    secrets_example = (PROJECT_ROOT / ".streamlit" / "secrets.toml.example").read_text(
        encoding="utf-8"
    )

    assert callback in readme
    assert "Allowed Callback URLs" in readme
    assert "Allowed Logout URLs" in readme
    assert "Allowed Logout URLs" in secrets_example


def _dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        assert separator, f"Invalid environment example line: {line}"
        values[key.strip()] = value.strip()
    return values
