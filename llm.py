from langchain_ollama.llms import OllamaLLM
from config import MODEL_NAME

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0,
)