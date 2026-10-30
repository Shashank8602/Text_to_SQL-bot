from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_host: str = "127.0.0.1"
    db_port: int = 5433
    app_db: str = "app"
    bot_reader_password: str
    rag_writer_password: str
    data_loader_password: str

    query_timeout_ms: int = 5000
    max_result_rows: int = 1000

    max_upload_mb: int = 200
    restore_timeout_s: int = 600

    # LLM providers. Model lists on free tiers change often: verify with the provider's
    # /models endpoint and override here via .env.
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    generator_model: str = "groq/openai/gpt-oss-120b"
    generator_fallback_model: str = "gemini/gemini-3.5-flash"
    judge_model: str = "groq/openai/gpt-oss-20b"
    judge_fallback_model: str = "gemini/gemini-3.5-flash-lite"
    # Free-tier limits, enforced by the router *before* calling, so it falls back instead
    # of hitting 429s. Groq reports 8000 tokens/min per model; keep some headroom.
    groq_rpm: int = 30
    groq_tpm: int = 7000
    gemini_rpm: int = 10
    llm_timeout_s: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
