from tools.web_search import WebSearchTool

from rag.splitter import split_documents
from rag.vectorstore import TemporaryVectorStore

from llm import llm
from prompts.system_prompt import SYSTEM_PROMPT
from utils import build_context


def answer_question(query: str) -> str:

    # Step 1: Collect documents
    web_tool = WebSearchTool()

    documents = web_tool.run(query)

    if not documents:
        return "No documents found."

    # Step 2: Chunk
    chunks = split_documents(documents)

    # Step 3: Create temporary vector DB
    store = TemporaryVectorStore()

    store.index(chunks)


    store.index(chunks)

    # Step 4: Retrieve
    docs = store.retrieve(query)

    # Step 5: Build prompt
    context = build_context(docs)

    chain = SYSTEM_PROMPT | llm

    return chain.invoke(
        {
            "question": query,
            "materials": context,
        }
    )


def main():

    while True:

        query = input("\nAsk a question (q to quit): ")

        if query.lower() == "q":
            break

        print(answer_question(query))


if __name__ == "__main__":
    main()