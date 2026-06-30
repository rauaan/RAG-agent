from typing import List

from ddgs import DDGS
from docling.document_converter import DocumentConverter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

MODEL_NAME = "gemma4:e2b"
EMBED_MODEL = "mxbai-embed-large"

MAX_RESULTS = 5

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

COLLECTION_NAME = "web_search"


# --------------------------------------------------------------------
# Initialize shared resources
# --------------------------------------------------------------------

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0,
)

converter = DocumentConverter()

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

prompt = ChatPromptTemplate.from_template(
"""
    Answer the question: {question}

    ONLY use the provided sources.

    {materials}

    Rules:

    - Every factual statement must have a supporting source.
    - List every unique source used.
    - If sources disagree, explicitly explain the disagreement.
    - Never invent information.

    Format:

    Answer:
    ...

    Sources:
    - https://...
    - https://...
"""
)


# --------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------

def search(query: str):
    return list(
        DDGS().text(
            query,
            max_results=MAX_RESULTS,
        )
    )


def fetch_documents(results) -> List[Document]:
    documents = []

    for result in results:
        url = result["href"]

        try:
            page = converter.convert(url)

            text = page.document.export_to_markdown()

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": url},
                )
            )

        except Exception as e:
            print(f"[WARN] Failed to fetch {url}")
            # print(e)

    return documents


def build_vector_store(documents: List[Document]):

    chunks = splitter.split_documents(documents)

    print(f"Embedding {len(chunks)} chunks...")

    for chunk in chunks:
        try:
            vector_store.add_documents([chunk])
        except Exception as e:
            print(
                f"[WARN] Skipping chunk from "
                f"{chunk.metadata.get('source')}"
            )
            print(e)

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20, 
            "lambda_mult": 0.5,
        },
    )


def build_context(docs: List[Document]) -> str:
    sections = []

    for doc in docs:
        sections.append(
            f"""SOURCE:
{doc.metadata['source']}

{doc.page_content}
"""
        )

    return "\n-----------------------------\n".join(sections)


def answer_question(query: str):

    print("Searching...")
    results = search(query)

    print("Fetching pages...")
    documents = fetch_documents(results)

    if not documents:
        return "No documents could be retrieved."

    print("Embedding...")
    retriever = build_vector_store(documents)

    print("Retrieving...")
    docs = retriever.invoke(query)

    context = build_context(docs)

    chain = prompt | llm

    print("Generating answer...")

    return chain.invoke(
        {
            "question": query,
            "materials": context,
        }
    )


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():

    while True:

        query = input("\nAsk a question (q to quit): ").strip()

        if query.lower() == "q":
            break

        try:
            response = answer_question(query)

            print("\n" + "=" * 80)
            print(response)
            print("=" * 80)

        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()