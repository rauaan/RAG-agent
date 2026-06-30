from langchain_chroma import Chroma

from rag.embeddings import embedding_model
from config import RETRIEVER_K, FETCH_K, LAMBDA_MULT


class TemporaryVectorStore:

    def __init__(self):

        self.store = Chroma(
            collection_name="temp",
            embedding_function=embedding_model,
        )

    def index(self, chunks):

        self.store.add_documents(chunks)

    def retrieve(self, query):

        retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": FETCH_K,
                "lambda_mult": LAMBDA_MULT,
            },
        )

        return retriever.invoke(query)