from pydantic import BaseModel
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
        print(result)


def test_docs_demo():
    from pydantic import BaseModel
    from capabilities import Capability

    # define the input and output models
    class InputText(BaseModel):
        text: str

    class WordTranslation(BaseModel):
        source: str
        target: str

    class TranslationOutput(BaseModel):
        translation: str
        word_translations: list[WordTranslation]

    # create an input instance
    inp = InputText(
        text="Understand: I'll slip quietly away through twilit meadows with only this one dream - you come too."
    )

    # provide some instructions
    instructions = "Given the input `text`, produce a `french_translation` which translates the `text` into French. Also produce a word-level translation called `word_translations`, which is a list of (english word, french transliteration) pairs."

    # print the task to console
    c = Capability("blazon/structured")
    translation_output: TranslationOutput = c(
        InputText, TranslationOutput, instructions, inp
    )
    print(translation_output)


def test_docs_demo2():
    from pydantic import BaseModel
    from capabilities import Capability

    class WordTranslation(BaseModel):
        source: str
        target: str

    class TranslationOutput(BaseModel):
        translation: str
        word_translations: list[WordTranslation]

    @llm
    def translate(input_text: str) -> TranslationOutput:
        """Given the input `text`, produces a `french_translation`
        which translates the `text` into French.
        Also produce a word-level translation called `word_translations`,
        which is a list of (english word, french transliteration) pairs.
        """
        ...

    # create an input instance
    inp = "Understand: I'll slip quietly away through twilit meadows with only this one dream - you come too."

    # print the task to console
    translation_output = translate(inp)
    print(translation_output)


if __name__ == "__main__":
    test_docs_demo2()
