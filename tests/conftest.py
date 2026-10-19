from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import DEFAULTS
from app.models import Base, RuntimeSetting, Setting


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        db.add(RuntimeSetting(id=1, execution_mode="STOPPED", entries_paused=True))
        db.add_all(Setting(key=key, value=value) for key, value in DEFAULTS.items())
        db.commit()
        yield db
    engine.dispose()
