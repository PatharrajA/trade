import json
from types import SimpleNamespace

from app import main
from app.models import ModelArtifact


def test_setup_activation_requires_completed_post_holdout_option_replay(session, monkeypatch):
    artifact = ModelArtifact(
        version="setup-gated", status="CANDIDATE", model_path="/tmp/setup-gated.json",
        metadata_path="/tmp/setup-gated.metadata.json",
        metadata_json=json.dumps({
            "version": "setup-gated", "model_type": "SETUP_QUALITY",
            "indices": ["NIFTY"], "validation_end": "2025-12-31",
            "activation_quality": {"ready": True, "issues": []},
        }),
    )
    session.add(artifact)
    session.commit()
    monkeypatch.setattr(main, "SetupQualityModel", lambda _path: SimpleNamespace(available=True))

    response = main.activate_setup_model(artifact.id, session)

    assert "replay+gate+blocked" in response.headers["location"]
    assert session.get(ModelArtifact, artifact.id).status == "CANDIDATE"
