"""The results of a training — metrics, curves, logs — uploaded to the loop beside its model.

A trainer names its result files in `_get_training_results`, grouped into parts. `upload_results` sends them
once the model is uploaded and never fails the training: a part the loop answered is done, whatever the
answer, a part that got no answer is sent up to `MAX_ATTEMPTS` times, and anything else that goes wrong is
logged and skips the part. `core` is always uploaded; `examples` and `logs` only if the training's
`upload_examples` and `upload_logs` hyperparameters ask for them. Each part goes up as one zip that
keeps the paths inside the training folder, and `core` additionally holds `meta.json`, where the
results come from.
"""
import asyncio
import json
import logging
import os
import zipfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import httpx

from ..data_classes import Training
from ..data_exchanger import DataExchanger
from .exceptions import TrainingResultsNotAnsweredError, TrainingResultsUploadFailedError
from .validation_set import bool_hyperparameter

CORE = 'core'
EXAMPLES = 'examples'
LOGS = 'logs'

UPLOAD_EXAMPLES = 'upload_examples'
UPLOAD_LOGS = 'upload_logs'
OPTIONAL_PARTS = {EXAMPLES: UPLOAD_EXAMPLES, LOGS: UPLOAD_LOGS}
"""The parts uploaded only if the hyperparameter they map to is true."""

META_FILE = 'meta.json'
RESULTS_FOLDER = 'training_results'
"""Inside the training folder: the zips and the record of the parts that are done."""
DONE_FILE = 'uploaded_parts.txt'

MAX_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds before the second attempt, doubled before each further one

logger = logging.getLogger(__name__)


def check_hyperparameters(hyperparameters: Mapping[str, Any]) -> None:
    """Raise a CriticalError if `upload_examples` or `upload_logs` is set to anything but a boolean."""
    for name in OPTIONAL_PARTS.values():
        bool_hyperparameter(hyperparameters, name)


async def upload_results(training: Training, *, model_uuid: str, node_name: str, node_uuid: str,
                         get_results: Callable[[Path], Mapping[str, list[Path]]], stop_requested: Callable[[], bool],
                         data_exchanger: DataExchanger) -> None:
    """Upload the parts of `get_results(training_folder)` the training's hyperparameters ask for; never raises.

    A part is recorded as done once it is answered or skipped, so a second call sends only the parts left.
    """
    training_folder = Path(training.training_folder)
    try:
        results = get_results(training_folder)
        parts = _parts_to_upload(results, training.hyperparameters)
        meta_content = _meta(training, model_uuid=model_uuid, node_name=node_name, node_uuid=node_uuid,
                             stop_requested=stop_requested())
    except Exception:
        logger.exception('Skipping the results of training %s', training.training_number)
        return
    results_folder = training_folder / RESULTS_FOLDER
    done = _load_done(results_folder)
    for part in [p for p in parts if p not in done]:
        try:
            await _upload_part(training, part, results[part], meta_content if part == CORE else None, data_exchanger)
        except TrainingResultsUploadFailedError as e:
            logger.warning('Skipping the %s results of training %s: %s', part, training.training_number, e)
        except Exception:
            logger.exception('Skipping the %s results of training %s', part, training.training_number)
        done.append(part)
        _save_done(results_folder, done)


def _parts_to_upload(results: Mapping[str, list[Path]], hyperparameters: Mapping[str, Any]) -> list[str]:
    """The parts of `results` to upload, in upload order: `core` first, the optional ones only if asked for."""
    parts = [part for part in results
             if part not in OPTIONAL_PARTS or bool_hyperparameter(hyperparameters, OPTIONAL_PARTS[part])]
    return sorted(parts, key=lambda part: part != CORE)


def _meta(training: Training, *, model_uuid: str, node_name: str, node_uuid: str,
          stop_requested: bool) -> dict[str, Any]:
    """Where the results come from, so they explain themselves without the loop's database."""
    return {
        'node_name': node_name,
        'node_uuid': node_uuid,
        'organization': training.context.organization,
        'project': training.context.project,
        'training_number': training.training_number,
        'model_uuid': model_uuid,
        'start_time': datetime.fromtimestamp(training.start_time, timezone.utc).isoformat(),
        'uploaded_at': datetime.now(timezone.utc).isoformat(),
        'stop_requested': stop_requested,
        'learning_loop_node_version': _library_version(),
        'hyperparameters': dict(training.hyperparameters),
    }


async def _upload_part(training: Training, part: str, files: list[Path], part_meta: dict[str, Any] | None,
                       data_exchanger: DataExchanger) -> None:
    """Zip and send one part; send it again while it gets no answer, up to `MAX_ATTEMPTS` times."""
    training_folder = Path(training.training_folder)
    zip_path = training_folder / RESULTS_FOLDER / f'{part}.zip'
    await asyncio.to_thread(_write_zip, files, training_folder, zip_path, part_meta)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            await data_exchanger.upload_training_results(training.context, training.training_number, part,
                                                         str(zip_path))
            return
        except (TrainingResultsNotAnsweredError, httpx.TransportError):
            if attempt == MAX_ATTEMPTS:
                raise
            await asyncio.sleep(RETRY_DELAY * 2 ** (attempt - 1))


def _write_zip(files: list[Path], training_folder: Path, zip_path: Path, meta: dict[str, Any] | None) -> None:
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


def _load_done(results_folder: Path) -> list[str]:
    path = results_folder / DONE_FILE
    return path.read_text().split() if path.exists() else []


def _save_done(results_folder: Path, done: list[str]) -> None:
    results_folder.mkdir(parents=True, exist_ok=True)
    (results_folder / DONE_FILE).write_text('\n'.join(done))


def _archive_name(path: Path, training_folder: Path) -> str:
    return str(path.relative_to(training_folder)) if path.is_relative_to(training_folder) else path.name


def _library_version() -> str | None:
    try:
        return version('learning_loop_node')
    except PackageNotFoundError:
        return None
