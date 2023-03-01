from capabilities.core import Capability
import fire

EXAMPLE_PASSAGE = """\
Wide-scale production is credited to Edward Goodrich Acheson in 1890.[11] Acheson was attempting to prepare artificial diamonds when he heated a mixture of clay (aluminium silicate) and powdered coke (carbon) in an iron bowl. He called the blue crystals that formed carborundum, believing it to be a new compound of carbon and aluminium, similar to corundum. Moissan also synthesized SiC by several routes, including dissolution of carbon in molten silicon, melting a mixture of calcium carbide and silica, and by reducing silica with carbon in an electric furnace. Acheson patented the method for making silicon carbide powder on February 28, 1893.[12] Acheson also developed the electric batch furnace by which SiC is still made today and formed the Carborundum Company to manufacture bulk SiC, initially for use as an abrasive.[13] In 1900 the company settled with the Electric Smelting and Aluminum Company when a judge's decision gave "priority broadly" to its founders "for reducing ores and other substances by the incandescent method".[14] It is said that Acheson was trying to dissolve carbon in molten corundum (alumina) and discovered the presence of hard, blue-black crystals which he believed to be a compound of carbon and corundum: hence carborundum. It may be that he named the material "carborundum" by analogy to corundum, which is another very hard substance (9 on the Mohs scale). The first use of SiC was as an abrasive. This was followed by electronic applications. In the beginning of the 20th century, silicon carbide was used as a detector in the first radios.[15] In 1907 Henry Joseph Round produced the first LED by applying a voltage to a SiC crystal and observing yellow, green and orange emission at the cathode. The effect was later rediscovered by O. V. Losev in the Soviet Union in 1923.[16]\
"""

def example():
    c = Capability("multi/document_qa")
    answer = c(EXAMPLE_PASSAGE, "Who formed the Carborundum Company?")
    print(answer)
    answer = c(EXAMPLE_PASSAGE, "What colors were emitted by the first LED crystal?")
    print(answer)

if __name__ == "__main__":
    fire.Fire(example)
