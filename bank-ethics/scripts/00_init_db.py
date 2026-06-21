import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bank_ethics.db.base import engine, Base
from bank_ethics.db import models  # noqa: F401

Base.metadata.create_all(bind=engine)
print("DB ready.")
