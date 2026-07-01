from enum import Enum

from llm import llm
from prompts.router_prompt import ROUTER_PROMPT


class Tool(Enum):
    LOCAL = "local_search"
    WEB = "web_search"
    BOTH = "both"
    NONE = "none"


class Router:

    def route(self, query: str) -> Tool:

        chain = ROUTER_PROMPT | llm

        decision = chain.invoke(
            {
                "question": query,
            }
        ).strip().lower()

        if decision == "local_search":
            return Tool.LOCAL

        if decision == "web_search":
            return Tool.WEB

        if decision == "both":
            return Tool.BOTH

        return Tool.NONE