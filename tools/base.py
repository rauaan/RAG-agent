from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseTool(ABC):

    name: str
    description: str

    @abstractmethod
    def run(self, query: str) -> list[Document]:
        pass