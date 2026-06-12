try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    from langchain.schema import Document

from src.ingestion import chunk_documents, load_documents


class FakeLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [
            Document(
                page_content=f"Contenu de {self.path}",
                metadata={"source": self.path, "page": 1},
            )
        ]


def test_load_documents_loads_pdf_and_markdown_files(tmp_path, monkeypatch):
    pdf_file = tmp_path / "test.pdf"
    md_file = tmp_path / "notes.md"
    txt_file = tmp_path / "ignored.txt"

    pdf_file.write_text("PDF content", encoding="utf-8")
    md_file.write_text("# Markdown content", encoding="utf-8")
    txt_file.write_text("Ignored content", encoding="utf-8")

    monkeypatch.setattr("src.ingestion.PyPDFLoader", FakeLoader)
    monkeypatch.setattr("src.ingestion.UnstructuredMarkdownLoader", FakeLoader)

    documents = load_documents(str(tmp_path))

    assert len(documents) == 2
    assert any(document.metadata["source"].endswith("test.pdf") for document in documents)
    assert any(document.metadata["source"].endswith("notes.md") for document in documents)
    assert not any(document.metadata["source"].endswith("ignored.txt") for document in documents)


def test_load_documents_returns_empty_list_when_no_supported_files(tmp_path):
    unsupported_file = tmp_path / "notes.txt"
    unsupported_file.write_text("Texte ignoré", encoding="utf-8")

    documents = load_documents(str(tmp_path))

    assert documents == []


def test_chunk_documents_splits_documents():
    content = " ".join(["phrase de test"] * 100)
    documents = [
        Document(
            page_content=content,
            metadata={"source": "document.md", "page": 1},
        )
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) > 1
    assert all(chunk.page_content for chunk in chunks)
    assert all(chunk.metadata["source"] == "document.md" for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)
