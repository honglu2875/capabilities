from capabilities import llm


def test_llm():
    @llm
    def fruit_emojis(n: int) -> list[str]:
        """Generates a list of fruit emojies of the given length.

        Example:
            >>> fruit_emojis(4) == ['🍎', '🍏', '🍊', '🍋']
        """
        ...

    for n in [0, 1, 2, 4, 10]:
        result = fruit_emojis(n)
        assert len(result) == n


if __name__ == "__main__":
    test_llm()
