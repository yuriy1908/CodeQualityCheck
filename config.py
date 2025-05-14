import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
    RAG_DB_PATH = os.getenv("RAG_DB_PATH", "data/rag_db")
    REPOS_ROOT = os.path.abspath(os.getenv("REPOS_ROOT", "data/repos"))