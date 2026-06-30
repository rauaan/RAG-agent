from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
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
""")