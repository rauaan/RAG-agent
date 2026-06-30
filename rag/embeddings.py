from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL

embedding_model = OllamaEmbeddings(
    model=EMBED_MODEL,
)