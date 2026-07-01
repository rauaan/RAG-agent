from langchain_community.document_loaders import PyPDFLoader
from rag.splitter import split_documents
from rag.vectorstore import PersistentVectorStore
from config import LOCAL_DOC_PATH


def main():

    store = PersistentVectorStore()

    loader = PyPDFLoader(LOCAL_DOC_PATH)

    documents = loader.load()

    chunks = split_documents(documents)

    store.index(chunks)

    print("Index built successfully.")


if __name__ == "__main__":
    main()