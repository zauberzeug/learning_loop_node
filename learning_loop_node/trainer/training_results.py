"""The results of a training — metrics, curves, logs — uploaded to the loop together with its model.

A trainer names its result files in `_get_training_results`, grouped into parts. The parts are uploaded
right after the model formats, in the same state, so every uploaded model gets its results. They are
optional: a part the loop refuses is skipped and the training goes on, and only a part that got no
answer is sent again. `core` is always uploaded; `examples` and `logs` only if the training's
`upload_examples` and `upload_logs` hyperparameters ask for them. Each part goes up as one zip that
keeps the paths inside the training folder, and `core` additionally holds `meta.json`, where the
results come from.
"""
import json
import logging
import os
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .validation_set import bool_hyperparameter

CORE = 'core'
EXAMPLES = 'examples'
LOGS = 'logs'

UPLOAD_EXAMPLES = 'upload_examples'
UPLOAD_LOGS = 'upload_logs'
OPTIONAL_PARTS = {EXAMPLES: UPLOAD_EXAMPLES, LOGS: UPLOAD_LOGS}
"""The parts uploaded only if the hyperparameter they map to is true."""

META_FILE = 'meta.json'

logger = logging.getLogger(__name__)


def parts_to_upload(results: Mapping[str, list[Path]], hyperparameters: Mapping[str, Any]) -> list[str]:
    """The parts of `results` to upload, in upload order: `core` first, the optional ones only if asked for."""
    parts = [part for part in results
             if part not in OPTIONAL_PARTS or bool_hyperparameter(hyperparameters, OPTIONAL_PARTS[part])]
    return sorted(parts, key=lambda part: part != CORE)


def write_zip(files: list[Path], training_folder: Path, zip_path: Path, meta: dict[str, Any] | None = None) -> None:
    """Zip `files` into `zip_path`, keeping their paths relative to `training_folder`; directories go in as a whole.

    A file that does not exist is logged and left out: a trainer may name results a stopped training never wrote.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(f'{zip_path.name}.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            if not file.exists():
                logger.warning('training result %s does not exist, leaving it out', file)
                continue
            for path in sorted(file.rglob('*')) if file.is_dir() else [file]:
                if path.is_file():
                    zipf.write(path, _archive_name(path, training_folder))
        if meta is not None:
            zipf.writestr(META_FILE, json.dumps(meta, indent=1))
    os.replace(partial, zip_path)


def _archive_name(path: Path, training_folder: Path) -> str:
    return str(path.relative_to(training_folder)) if path.is_relative_to(training_folder) else path.name


def meta(*, node_name: str, node_uuid: str, organization: str, project: str, training_number: int, model_uuid: str,
         start_time: float, stop_requested: bool, hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    """Where the results come from, so they explain themselves without the loop's database."""
    return {
        'node_name': node_name,
        'node_uuid': node_uuid,
        'organization': organization,
        'project': project,
        'training_number': training_number,
        'model_uuid': model_uuid,
        'start_time': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        'uploaded_at': datetime.now(timezone.utc).isoformat(),
        'stop_requested': stop_requested,
        'learning_loop_node_version': _library_version(),
        'hyperparameters': dict(hyperparameters),
    }


def _library_version() -> str | None:
    try:
        return version('learning_loop_node')
    except PackageNotFoundError:
        return None
