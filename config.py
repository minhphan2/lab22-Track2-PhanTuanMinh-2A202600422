"""Shared configuration — loads .env and exposes constants."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# LangSmith
LANGSMITH_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "lab22")

# OpenAI-compatible
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL       = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

if __name__ == "__main__":
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {LANGSMITH_PROJECT}")
    print(f"   OpenAI endpoint   : {OPENAI_BASE_URL}")
    print(f"   Default LLM model : {LLM_MODEL}")
    print(f"   Embedding model   : {EMBEDDING_MODEL}")