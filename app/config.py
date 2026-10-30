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


@lru_cache
def get_settings() -> Settings:
    return Settings()
