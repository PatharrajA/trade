from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ModelArtifact
from .strategy import RegimeModel, SUPPORTED_FEATURE_SCHEMAS
from .training import candidate_quality_issues, load_model_metadata, model_metadata_path


def _remove_artifact_files(artifact: ModelArtifact) -> None:
    """Remove a manifest and any model components declared by its metadata."""
    try:
        metadata = json.loads(artifact.metadata_json)
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    parent = Path(artifact.model_path).parent
    for detail in (metadata.get("index_models") or {}).values():
        component = Path(str(detail.get("file") or ""))
        if component.name and component.name == str(component):
            (parent / component.name).unlink(missing_ok=True)
    Path(artifact.model_path).unlink(missing_ok=True)
    Path(artifact.metadata_path).unlink(missing_ok=True)


def register_candidate(session: Session, model_path: Path, metadata: dict) -> ModelArtifact:
    artifact = ModelArtifact(
        version=str(metadata["version"]),
        status="CANDIDATE",
        model_path=str(model_path),
        metadata_path=str(model_metadata_path(model_path)),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    session.add(artifact)
    session.commit()
    candidates = session.scalars(
        select(ModelArtifact)
        .where(ModelArtifact.status == "CANDIDATE")
        .order_by(ModelArtifact.created_at.desc())
    ).all()
    for obsolete in candidates[5:]:
        obsolete.status = "PRUNED"
        _remove_artifact_files(obsolete)
    session.commit()
    return artifact


def activate_candidate(session: Session, artifact_id: int, active_path: Path) -> ModelArtifact:
    artifact = session.get(ModelArtifact, artifact_id)
    if artifact is None:
        raise ValueError("Model candidate was not found")
    source = Path(artifact.model_path)
    source_metadata = Path(artifact.metadata_path)
    metadata = load_model_metadata(source)
    if not source.exists() or not source_metadata.exists() or metadata.get("feature_schema") not in SUPPORTED_FEATURE_SCHEMAS:
        raise ValueError("Candidate files are missing or use an incompatible feature schema")
    if not RegimeModel(source).available:
        raise ValueError("Candidate model cannot be loaded")
    quality_issues = candidate_quality_issues(metadata)
    if quality_issues:
        raise ValueError("Candidate failed the activation quality gate: " + "; ".join(quality_issues))

    active_path.parent.mkdir(parents=True, exist_ok=True)
    archive = active_path.parent / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    current = load_model_metadata(active_path)
    if active_path.exists() and current.get("version"):
        archived_model = archive / f"{current['version']}.json"
        archived_metadata = archive / f"{current['version']}.metadata.json"
        shutil.copy2(active_path, archived_model)
        shutil.copy2(model_metadata_path(active_path), archived_metadata)

    temporary_model = active_path.with_name(f".{active_path.name}.activation")
    temporary_metadata = model_metadata_path(active_path).with_name(
        f".{model_metadata_path(active_path).name}.activation"
    )
    shutil.copy2(source, temporary_model)
    shutil.copy2(source_metadata, temporary_metadata)
    os.replace(temporary_model, active_path)
    os.replace(temporary_metadata, model_metadata_path(active_path))

    for existing in session.scalars(select(ModelArtifact).where(ModelArtifact.status == "ACTIVE")):
        existing.status = "ARCHIVED"
    artifact.status = "ACTIVE"
    artifact.activated_at = datetime.now(UTC)
    session.commit()

    archived = session.scalars(
        select(ModelArtifact)
        .where(ModelArtifact.status == "ARCHIVED")
        .order_by(ModelArtifact.activated_at.desc(), ModelArtifact.created_at.desc())
    ).all()
    for obsolete in archived[5:]:
        obsolete.status = "PRUNED"
        _remove_artifact_files(obsolete)
        (archive / f"{obsolete.version}.json").unlink(missing_ok=True)
        (archive / f"{obsolete.version}.metadata.json").unlink(missing_ok=True)
    session.commit()
    return artifact
