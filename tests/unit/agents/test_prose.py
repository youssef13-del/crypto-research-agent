from crypto_research.shared.text import clean_generated_text


def test_clean_generated_text_removes_card_markup_and_duplicate_sentences() -> None:
    value = (
        "### **Analysis:** The verified event changes the project's operating context. "
        "The verified event changes the project's operating context."
    )

    assert (
        clean_generated_text(
            value,
            max_chars=240,
            ensure_sentence=True,
        )
        == "The verified event changes the project's operating context."
    )


def test_clean_generated_text_compacts_without_cutting_a_word() -> None:
    value = "Summary: " + "meaningful context " * 30

    result = clean_generated_text(value, max_chars=90, ensure_sentence=True)

    assert len(result) <= 90
    assert result.endswith(".")
    assert "Summary:" not in result
