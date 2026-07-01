from router import Router, Tool

from tools.local_search import LocalSearchTool
from tools.web_search import WebSearchTool

from rag.splitter import split_documents
from rag.vectorstore import TemporaryVectorStore

from llm import llm
from prompts.system_prompt import SYSTEM_PROMPT
from utils import build_context


router = Router()

local_tool = LocalSearchTool()
web_tool = WebSearchTool()


def answer_question(query: str):

    decision = router.route(query)

    documents = []

    if decision == Tool.LOCAL:
        print("Using local search...")
        documents.extend(local_tool.run(query))

    elif decision == Tool.WEB:
        print("Using web search...")
        documents.extend(web_tool.run(query))

    elif decision == Tool.BOTH:
        print("Using local + web search...")
        documents.extend(local_tool.run(query))
        documents.extend(web_tool.run(query))

    else:
        return "I don't know which tool to use."

    if not documents:
        return "No documents found."

    chunks = split_documents(documents)

    store = TemporaryVectorStore()

    store.index(chunks)

    docs = store.retrieve(query)

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

        query = input("\nAsk a question (q to quit): ").strip()

        if query.lower() == "q":
            break

        print(answer_question(query))


if __name__ == "__main__":
    main()