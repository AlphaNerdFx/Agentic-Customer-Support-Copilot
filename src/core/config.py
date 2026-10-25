"""Application settings and environment configuration."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM & Embeddings
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Primary chat model")
    embedding_model: str = Field(
        default="text-embedding-3-small", 
        description="Embedding model"
    )

    # Storage Paths
    chroma_persist_dir: str = Field(
        default="./data/chroma", 
        description="Persistent directory for ChromaDB"
    )
    sqlite_db_path: str = Field(
        default="./data/checkpoints.db", 
        description="SQLite database for LangGraph state checkpointer"
    )

    # Observability
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str = Field(default="fde-copilot")
    log_level: str = Field(default="INFO")


settings = Settings()
