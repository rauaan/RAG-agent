from langchain_core.documents import Document

def build_context(
    documents: list[Document]
) -> str:

    sections = []

    for doc in documents:

        sections.append(
            f"""SOURCE:
{doc.metadata["source"]}

{doc.page_content}
"""
        )

    return "\n---------------------\n".join(sections)