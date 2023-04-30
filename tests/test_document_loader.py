from capabilities.search.loader import create_document


def test_webpdf():
    create_document("https://arxiv.org/pdf/2303.16200.pdf")
    print("parsed web pdf")


def test_html():
    create_document(
        "https://gowers.wordpress.com/2022/04/28/announcing-an-automatic-theorem-proving-project/#more-6531",
    )
    print("parsed web html")


def test_localpdf():
    create_document("examples/data/sample.pdf")
    print("parsed local pdf")


def test_md():
    doc = create_document("examples/data/sample.md")
    print("parsed local text")
