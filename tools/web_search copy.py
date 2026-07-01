from ddgs import DDGS
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from config import MAX_WEB_RESULTS
from .base import BaseTool


class WebSearchTool(BaseTool):

    name = "web_search"

    description = (
        "Search the internet for recent or external information "
        "and return documents."
    )

    def __init__(self):
        self.converter = DocumentConverter()

    def search(self, query: str):
        return list(
            DDGS().text(
                query,
                max_results=MAX_WEB_RESULTS,
            )
        )

    def run(self, query: str) -> list[Document]:

        documents = []

        results = self.search(query)

        for result in results:

            url = result["href"]

            try:
                page = self.converter.convert(url)

                text = page.document.export_to_markdown()

                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url},
                    )
                )

            except Exception:
                print(f"[WARN] Failed to fetch {url}")

        return documents