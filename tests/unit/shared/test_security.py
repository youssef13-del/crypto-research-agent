from datetime import UTC, datetime

from crypto_research.domain.research import (
    FundamentalEvidence,
    NewsItem,
)
from crypto_research.shared.security import (
    escape_markdown,
    normalize_http_url,
    normalize_user_question,
    redact_secrets,
)


def test_redact_secrets_handles_prefixed_and_quoted_values() -> None:
    value = 'GROQ_API_KEY=first {"api_key": "second"} authorization=\'Bearer third\' Bearer fourth'

    redacted = redact_secrets(value)

    assert "first" not in redacted
    assert "second" not in redacted
    assert "third" not in redacted
    assert "fourth" not in redacted
    assert "GROQ_API_KEY=[redacted]" in redacted
    assert '"api_key": "[redacted]"' in redacted


def test_redact_secrets_consumes_spaces_inside_quoted_values() -> None:
    value = '"GROQ_API_KEY": "first value" \'access_token\'=\'second value\' Bearer "third value"'

    redacted = redact_secrets(value)

    assert "first value" not in redacted
    assert "second value" not in redacted
    assert "third value" not in redacted
    assert '"GROQ_API_KEY": "[redacted]"' in redacted
    assert "'access_token'='[redacted]'" in redacted
    assert "Bearer [redacted]" in redacted


def test_redact_secrets_consumes_escaped_quotes_inside_values() -> None:
    value = r'{"GROQ_API_KEY": "first\" private"} Bearer "second\" private"'

    redacted = redact_secrets(value)

    assert "private" not in redacted
    assert '"GROQ_API_KEY": "[redacted]"' in redacted
    assert "Bearer [redacted]" in redacted


def test_redact_secrets_preserves_crypto_token_assignment() -> None:
    value = "Research token=ETH but use api_token=private-value"

    redacted = redact_secrets(value)

    assert "token=ETH" in redacted
    assert "private-value" not in redacted
    assert "api_token=[redacted]" in redacted


def test_redact_secrets_is_idempotent() -> None:
    values = (
        "GROQ_API_KEY=sk-1234567890",
        '{"api_key": "second"}',
        "authorization='Bearer third'",
        "Authorization: Bearer sk-abcd",
        "Bearer fourth",
    )

    for value in values:
        redacted = redact_secrets(value)
        assert redact_secrets(redacted) == redacted


def test_normalize_user_question_collapses_whitespace_and_bounds_length() -> None:
    normalized = normalize_user_question("  \r\n Compare   Bitcoin\r\n\r\n and Ethereum.  ")

    assert normalized == "Compare Bitcoin and Ethereum."
    assert len(normalize_user_question("x" * 5000)) == 2000


def test_normalize_user_question_drops_null_bytes() -> None:
    assert normalize_user_question("hello\x00 world") == "hello world"


def test_normalize_http_url_removes_sensitive_parts() -> None:
    url = "HTTPS://Example.COM/story?q=btc&GROQ_API_KEY=hidden&X-Amz-Signature=secret#private"

    assert normalize_http_url(url) == "https://example.com/story?q=btc"


def test_normalize_http_url_rejects_unsafe_destinations() -> None:
    values = (
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://user:password@example.com/story",
        "http://127.0.0.1/private",
        "http://[::1]/private",
        "https://localhost/private",
        "http://localhost./private",
        "http://127.1/private",
        "http://0177.0.0.1/private",
        "http://0x7f.0.0.1/private",
        "https://metadata/private",
        "https://service.internal/private",
        "https://example.com/a path",
        "https://example.com\\redirect",
    )

    assert all(normalize_http_url(value) is None for value in values)


def test_escape_markdown_escapes_untrusted_labels() -> None:
    label = "Publisher:\n# ](javascript:alert(1)) *headline*"

    escaped = escape_markdown(label)

    assert r"\]" in escaped
    assert "\n" not in escaped
    assert r"\#" in escaped
    assert r"\*headline\*" in escaped


def test_surviving_evidence_models_discard_unsafe_provider_urls() -> None:
    news = NewsItem(
        publisher="Example",
        title="Unsafe link",
        excerpt="Provider supplied content",
        url="javascript:alert(1)",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    fundamentals = FundamentalEvidence(homepage="http://127.0.0.1/private")

    assert news.url is None
    assert fundamentals.homepage is None
