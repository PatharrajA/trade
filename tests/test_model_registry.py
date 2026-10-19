import json

from app.model_registry import activate_candidate, register_candidate
from app.models import ModelArtifact
from app.strategy import FEATURE_SCHEMA


class _LoadableModel:
    def __init__(self, _path):
        self.available = True


def test_candidate_requires_explicit_activation_and_is_copied_atomically(session, tmp_path, monkeypatch):
    candidate = tmp_path / "candidates" / "candidate.json"
    candidate.parent.mkdir()
    candidate.write_text("model-bytes")
    metadata = {"version": "regime-test", "feature_schema": FEATURE_SCHEMA}
    candidate.with_name("candidate.metadata.json").write_text(json.dumps(metadata))

    artifact = register_candidate(session, candidate, metadata)
    assert artifact.status == "CANDIDATE"

    monkeypatch.setattr("app.model_registry.RegimeModel", _LoadableModel)
    active = tmp_path / "regime.json"
    activate_candidate(session, artifact.id, active)

    session.refresh(artifact)
    assert artifact.status == "ACTIVE"
    assert active.read_text() == "model-bytes"
    assert json.loads(active.with_name("regime.metadata.json").read_text())["version"] == "regime-test"
    assert session.query(ModelArtifact).count() == 1
