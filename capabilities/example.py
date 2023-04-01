import fire
from capabilities import Capability
from pydantic import BaseModel
from typing import List

EXAMPLE_PASSAGE = """\
Wide-scale production is credited to Edward Goodrich Acheson in 1890.[11] Acheson was attempting to prepare artificial diamonds when he heated a mixture of clay (aluminium silicate) and powdered coke (carbon) in an iron bowl. He called the blue crystals that formed carborundum, believing it to be a new compound of carbon and aluminium, similar to corundum. Moissan also synthesized SiC by several routes, including dissolution of carbon in molten silicon, melting a mixture of calcium carbide and silica, and by reducing silica with carbon in an electric furnace. Acheson patented the method for making silicon carbide powder on February 28, 1893.[12] Acheson also developed the electric batch furnace by which SiC is still made today and formed the Carborundum Company to manufacture bulk SiC, initially for use as an abrasive.[13] In 1900 the company settled with the Electric Smelting and Aluminum Company when a judge's decision gave "priority broadly" to its founders "for reducing ores and other substances by the incandescent method".[14] It is said that Acheson was trying to dissolve carbon in molten corundum (alumina) and discovered the presence of hard, blue-black crystals which he believed to be a compound of carbon and corundum: hence carborundum. It may be that he named the material "carborundum" by analogy to corundum, which is another very hard substance (9 on the Mohs scale). The first use of SiC was as an abrasive. This was followed by electronic applications. In the beginning of the 20th century, silicon carbide was used as a detector in the first radios.[15] In 1907 Henry Joseph Round produced the first LED by applying a voltage to a SiC crystal and observing yellow, green and orange emission at the cathode. The effect was later rediscovered by O. V. Losev in the Soviet Union in 1923.[16]\
"""


def example_structured():
    # structured synthesis capability
    # callable and accepts Pydantic dataclasses for specification
    # along with a natural language instruction
    c = Capability("multi/structured")

    class Document(BaseModel):
        """
        A class representing a document with a text field.
        """

        text: str

    # informal spec: bullet_point clearly follows from the supporting text
    class SupportedBulletPoint(BaseModel):
        """
        A class representing a supported bullet point in a document summary.
        A supported bullet point has a bullet point summarizing a passage from the text,
        as well as the supporting text itself, which is a passage of text extracted verbatim from the document.
        """

        bullet_point: str
        supporting_text: str

    class DocumentSummary(BaseModel):
        """
        A class representing a summary of a document, consisting of a list of supported bullet points.
        """

        supportedBulletPoints: List[SupportedBulletPoint]

    # other part of the informal spec
    instructions: str = """\
Write a bullet-pointed summary of the `text` as a list of supported bullet points, each with a `bullet_point` summarizing a passage from the `text` called `supporting_text`. Each `supporting_text` should be a passage of text extracted verbatim from `text` which supports its corresponding `bullet_point`. The list should contain no more than five (5) items.
    """

    inp = Document(text=EXAMPLE_PASSAGE)

    # synthesis call
    document_summary = Capability("multi/structured")(
        input_spec=Document, output_spec=DocumentSummary, input=inp, instructions=instructions
    )

    for x in document_summary.supportedBulletPoints:
        print(f"claim: {x.bullet_point}")
        print(f'    support: """{x.supporting_text}"""\n')


def example_document_qa():
    c = Capability("multi/document_qa")
    question = "Who formed the Carborundum Company?"
    answer = c(EXAMPLE_PASSAGE, question)
    print(dict(question=question))
    print(answer)
    question = "What colors were emitted by the first LED crystal?"
    answer = c(EXAMPLE_PASSAGE, question)
    print(dict(question=question))
    print(answer)


def example_search():
    c = Capability("multi/search")
    print(c("Who discovered the useful properties of silicon carbide?"))


def parse_table():
    class UnstructuredTable(BaseModel):
        unstructured_table: str

    class ParsedCell(BaseModel):
        cell_content: str

    class ParsedRow(BaseModel):
        cells: List[ParsedCell]

    class ParsedTable(BaseModel):
        rows: List[ParsedRow]

    instructions: str = """\
    Given the following `unstructured_table` ASCII representation of a table, parse it into a list of rows where each row is a list of cells, consisting of the contents of each cell of the `unstructured_table` with all trailing and leading whitespace stripped."""

    c = Capability("multi/structured")

    unstructured_table = """\
|mathqa        |      0|acc     |  0.2500|_  | 0.0435|
|              |       |acc_norm|  0.2300|_  | 0.0423|
|lambada_openai|      0|ppl     |145.0963|_  |53.1010|
|              |       |acc     |  0.2400|_  | 0.0429|
|boolq         |      1|acc     |  0.6000|_  | 0.0492|
|hellaswag     |      0|acc     |  0.3000|_  | 0.0461||
              |       |acc_norm|  0.3600|_  | 0.0482|
|triviaqa      |      1|acc     |  0.0000|_  | 0.0000|
|winogrande    |      0|acc     |  0.4000|_  | 0.0492|
|race          |      1|acc     |  0.2900|_  | 0.0456|"""

    parsed_table = c(
        UnstructuredTable,
        ParsedTable,
        instructions,
        UnstructuredTable(unstructured_table=unstructured_table),
    )

    for row in parsed_table.rows:
        print("|".join(cell.cell_content for cell in row.cells))


def test_factorization():
    class Number(BaseModel):
        value: int

    class PrimeFactor(BaseModel):
        prime: int
        exponent: int

    class PrimeFactorization(BaseModel):
        factors: List[PrimeFactor]

    instructions: str = """\
    Given the `value: int`, return a prime factorization as a list of pairs of `prime` and `exponent`s."""

    def verify(input, factorization):
        running_product = 1
        for factor in factorization.factors:
            running_product *= factor.prime**factor.exponent

        print(f"running_product={running_product}")
        return input.value == running_product

    c = Capability("multi/structured")

    import random

    for x in random.choices(range(150, 550), k=4):
        input = Number(value=x)
        output = c(Number, PrimeFactorization, instructions, input)
        print(output)
        print(verify(input, output))


def structured_chain_of_thought():
    class Input(BaseModel):
        input: str

    class ChainOfThought(BaseModel):
        thoughts: List[List[str]]
        conclusion: str
        meta_conclusion: str

    class CombinedInputOutput(BaseModel):
        input: Input
        cot: ChainOfThought

    class Output(BaseModel):
        output: str

    c = Capability("multi/structured")

    def generate_chain_of_thought(input: Input) -> ChainOfThought:
        cot_instructions = """\
Let's think step by step. Generate a list of `thought`s which comprise a detailed and correct solution to the problem posed in the `input`. Each member of `thoughts` should be a list with two strings, where the first one is a step in the solution and the second one is a reflection on whether the first one is correct or incorrect. Then, based on the `thoughts`, write a sentence-long `conclusion`. Then also add a `meta_conclusion` which explains why the `conclusion` follows from the `thought`s.
        """
        return c(Input, ChainOfThought, cot_instructions, input)

    def generate_output(input, chain_of_thought) -> Output:
        output_instructions = """\
After thinking step by step and reaching a conclusion, summarize the conclusion as a standalone `output`.
        """
        combined = CombinedInputOutput(input=input, cot=chain_of_thought)
        return c(CombinedInputOutput, Output, output_instructions, combined)

    input = Input(input="""What is the prime factorization of 7975?""")
    cot = generate_chain_of_thought(input)
    print("COT: ", cot)
    output = generate_output(input, cot)
    print("OUTPUT: ", output)

    # print(sr("yeehaw"))
    # import urllib.parse
    # query_str = urllib.parse.quote(query)
    # search_results_page = wc(f"https://www.google.com/search?q={query_str}")
    # print("SEARCH RESULTS: ", search_results_page)


def example_translation():
    """
    Translate some text into French.
    """

    # define the input and output models
    class InputText(BaseModel):
        text: str

    class WordTranslation(BaseModel):
        source: str
        target: str

    class TranslationOutput(BaseModel):
        translation: str
        word_translations: List[WordTranslation]

    # create an input instance
    inp = InputText(
        text="Understand: I'll slip quietly away through twilit meadows with only this one dream - you come too."
    )

    # provide some instructions
    instructions = "Given the input `text`, produce a `french_translation` which translates the `text` into French. Also produce a word-level translation called `word_translations`, which is a list of (english word, french transliteration) pairs."

    # print the task to console
    result = Capability("multi/structured")(InputText, TranslationOutput, instructions, inp)
    import json

    print(json.dumps(result.dict(), indent=2))


def example_summarize():
    print(Capability("multi/summarize")(EXAMPLE_PASSAGE))


# example usage: python -m capabilities.example example_translation
if __name__ == "__main__":
    import sys

    fire.Fire(component=locals()[sys.argv[1]], command=sys.argv[2:])
