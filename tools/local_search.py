from .base import BaseTool
from langchain_core.documents import Document
from rag.vectorstore import PersistentVectorStore

class LocalSearchTool:
    

    name = "local_search"

    description = (
        "Search the local documents for information "
        "and return documents."
    )


    def __init__(self):
        self.store = PersistentVectorStore()

    def run(self, query: str) -> list[Document]:
        return self.store.retrieve(query)