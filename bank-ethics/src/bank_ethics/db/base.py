import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def _resolve_database_url(raw_url: str) -> str:
	sqlite_prefix = "sqlite:///"

	if not raw_url.startswith(sqlite_prefix):
		return raw_url

	db_path = raw_url[len(sqlite_prefix) :]
	candidate = Path(db_path)

	if candidate.is_absolute():
		resolved = candidate
	else:
		resolved = (PROJECT_ROOT / candidate).resolve()

	resolved.parent.mkdir(parents=True, exist_ok=True)
	return f"{sqlite_prefix}{resolved.as_posix()}"

DATABASE_URL = _resolve_database_url(
	os.getenv("DATABASE_URL", "sqlite:///data/bank_ethics.db")
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()
