# AI Agent Guidelines for the Learning Loop Node Library

`learning_loop_node` is the **public Python library** every node uses to talk to the
[Learning Loop](https://learning-loop.ai). Four node types build on it: Trainer, Detector,
Annotator and Converter. It is published to PyPI, so its public surface is an API that the
Learning Loop backend and every node repository depends on — `../yolov5_node` is the public one.

For coding standards see [CONTRIBUTING.md](CONTRIBUTING.md). [README.md](README.md) documents the
environment variables, the node types and how to write a node against them.

## Layout

- `learning_loop_node/` — the library itself: `node.py` and `rest.py` for the shared node
  machinery, `trainer/`, `detector/`, `annotation/` for the per-type base logic, `data_classes/`
  and `enums/` for the wire types, `loop_communication.py` and `data_exchanger.py` for the
  loop-facing HTTP and socket.io traffic.
- `learning_loop_node/tests/` — `annotator`, `detector`, `trainer` and `general` suites.
- `mock_trainer/`, `mock_detector/`, `mock_annotator/` — reference implementations with their own
  tests. They are what `loop`'s CI runs against, so they are also the best template for a new node.
- `demo_segmentation_tool/` — a worked annotator example.

## Architecture

Every node is a `FastAPI` subclass (`node.py`): its lifespan connects to the loop and starts a
`repeat_loop` that calls the subclass' `on_repeat` every `repeat_loop_cycle_sec` (5 s) — that loop,
not an event handler, is what drives status reporting, model updates and training continuation.
Subclasses implement `on_startup`, `on_shutdown`, `on_repeat` and `register_sio_events`.

Nodes communicate with the loop via one or both of two channels:
- `LoopCommunicator` (httpx and login cookies) for the REST API
-  A socket.io client for status updates and loop-issued commands.
A `DataExchanger` sits on top of the communicator to move images and model zips

- **Trainer** — `TrainerLogicGeneric._training_loop` is a state machine over `TrainerState`
  (`enums/trainer.py`): download data → download base model → train → sync confusion matrix →
  upload model → detect → upload detections → cleanup. `_perform_state` wraps each step: an
  ordinary exception records the error and rewinds to the previous state (retried on the next
  cycle), a `CriticalError` jumps to `ReadyForCleanup`. Every transition is persisted through
  `LastTrainingIO`, so `try_continue_run_if_incomplete` resumes an interrupted training after a
  restart. The loop starts a training via the `begin_training` sio event. A concrete trainer
  implements `_train`, `_do_detections`, `_get_new_best_training_state`, `_on_metrics_published`,
  `_get_latest_model_files` and `_clear_training_data`; `TrainerLogic` adds an `Executor` for
  trainers that shell out to a training process.
- **Detector** — Detectors are rolled out on user machines and thus kept very simple -> `needs_login=False, needs_sio=False`. In the loop have few non-destructive capabilities and sio has shown to cause traffic spikes when trying to reconnect on bad connections.
  It *hosts* a socket.io server for its own clients and polls
  `/{org}/projects/{project}/deployment/target` over REST in `on_repeat` instead. `_DetectorState`
  (`_Initializing` / `_Updating` / `_ActiveDetector`) models the model swap: download to
  `models/<version>`, build a `DetectorLogic` through the factory, then swap atomically so the old
  model keeps serving until the new one is ready (unless `EXCLUSIVE_MODEL_BUILD` frees VRAM first).
  `OperationMode` gates whether updates may happen at all. Detections flow through
  `RelevanceFilter`, which writes selected images to the `Outbox` on disk; a separate upload process
  drains it.
- **Annotator** — thin: it forwards the loop frontend's `handle_user_input` events into
  `AnnotatorLogic` and keeps a per-frontend history.

`trainer/subprocess.py`, `trainer/batch_size.py` and `trainer/metrics.py` hold the parts of a
trainer that are not framework-specific. `iterator_cpu_bound` runs a training generator in a
spawned process and yields its progress through a `maxsize=1` queue, so the event loop stays
responsive, CUDA state stays out of the node process, and the training can never run more than
one item ahead of the bookkeeping. `find_batch_size` probes for the largest power-of-two batch
that fits, around a `fits` predicate the trainer supplies — none of it imports torch, so a node
brings its own way of running a step. `macro_f1` scores the confusion matrix
`_get_new_best_training_state` returns.

`helpers/entrypoint.py` holds what every node's `main.py` repeats: `node_parser` builds a
configargparse parser with `--host`/`--port`, `run_node` starts uvicorn. A setting is a flag
*and* an environment variable from one declaration — `--conf-threshold` reads
`CONF_THRESHOLD`. The exception is `--host`/`--port`, which read `NODE_HOST`/`NODE_PORT`: the
bare `HOST` already means the loop's address, and a node adopting it would hand it to uvicorn
and fail to bind. A node that used to require a prefix passes `legacy_env_prefix`, and the
prefixed names keep working with a warning.

`detector/postprocess.py` and `detector/geometry.py` hold the parts of a detector that do *not*
depend on the model: confidence filtering, per-class NMS, box/point clipping, and turning
predictions into the loop's dataclasses. A node should import them rather than write its own —
every node repository had grown its own drifting copy, which is why they live here. Note the two
containers: `to_image_metadata` builds what a **detector** node reports, `to_detections` what a
**trainer**'s auto-detection pass reports. Both go through one routine, so the two paths cannot
drift apart again.

All node state lives under `GLOBALS.data_folder` (`DATA_FOLDER`, default `/data`): `uuids.json`
(the node uuid is derived from its name and reused across restarts), `models/` plus the
`current_model` symlink, `outbox/`, and the per-project training folders.

## Running and testing

The `unit` suite is self-contained — no Learning Loop, no credentials, no network — so it is the
one to run while iterating, and the one CI gates the others on:

```bash
python -m pytest learning_loop_node/tests/unit -v
```

Every other suite talks to a real Learning Loop instance and reads its credentials from a local
`.env` (`LOOP_HOST`, `LOOP_USERNAME`, `LOOP_PASSWORD`). Without a reachable loop those cannot pass —
do not treat their failure as a regression you introduced.

```bash
./run_tests.sh              # all suites
./run_tests.sh <filter>     # passed to pytest as -k
```

Each suite carries its own `pytest.ini` (that is where `asyncio_mode = auto` comes from), so always
run pytest with a path inside one suite — a bare `pytest` from the repository root picks up no
config and the async tests error out:

```bash
python -m pytest learning_loop_node/tests/trainer -v                     # one suite
python -m pytest learning_loop_node/tests/trainer/test_errors.py -v      # one file
python -m pytest learning_loop_node/tests/trainer -v -k <test_name>      # one test
```

An autouse fixture repoints `GLOBALS.data_folder` at `/tmp/learning_loop_lib_data` and wipes it
around every test, so tests never touch `/data`. The `general` suite generates and deletes a real
`zauberzeug/pytest_nodelib_general` project on the loop; the detector suite starts the node in a
forked uvicorn process on `GLOBALS.detector_port`.

There is no `.pre-commit-config.yaml` here and no ruff in the project environment. Lint with:

```bash
uvx ruff check .
```

A clean tree already reports several hundred ruff findings, so a clean run is not a reachable goal.
Compare the count on the files you touched, before and after.

`.github/workflows/pytest.yml` runs the suites (the `unit` job first, without secrets),
`publish.yml` releases to PyPI on a tagged release.

## Working in this repository

- **This is a library — declaring a dependency is part of its API.** Before removing or loosening
  one, grep every consuming repository checked out beside this one for the package: a consumer that
  imports it without declaring it inherits it from here and breaks when it goes away.
- **Renaming or reshaping anything exported** breaks those repositories. Say so in the pull request
  and check whether a companion change is needed there.
- Bump `version` in `pyproject.toml` for a release; the trainer nodes pin the library version in
  their image tags (`A.B.C-nlvX.Y.Z`).
