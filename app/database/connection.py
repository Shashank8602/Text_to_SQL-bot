from typing import Literal

from psycopg.conninfo import make_conninfo

from app.config import get_settings

Role = Literal["bot_reader", "rag_writer", "data_loader"]


def conninfo(role: Role, dbname: str) -> str:
    """Connection string for one of the three app roles (see db/init/01_setup.sh)."""
    s = get_settings()
    password = {
        "bot_reader": s.bot_reader_password,
        "rag_writer": s.rag_writer_password,
        "data_loader": s.data_loader_password,
    }[role]
    return make_conninfo(
        host=s.db_host, port=s.db_port, user=role, password=password,
        dbname=dbname, connect_timeout=5,
    )


def app_conninfo() -> str:
    return conninfo("rag_writer", get_settings().app_db)
