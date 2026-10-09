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
  trainers that shell out to a training process. `trainer/training_slot.py` is the right to use
  the GPU when several trainers share one (`TRAINING_SLOT_LOCK`, an `flock` on a mounted file):
  `_run` acquires it before the state machine and releases it after, `state` reports `blocked` while
  a sibling holds it and there is no training. A training waiting for the slot reports
  `waiting_for_slot`, never `blocked`: the loop ends a training on `blocked` just as on `idle`,
  because a waiting sibling usually grabs the freed slot before the finished trainer's next status
  report. Without the variable the slot is always free.
- **Detector** — Detectors are rolled out on user machines and thus kept very simple -> `needs_login=False, needs_sio=False`. In the loop have few non-destructive capabilities and sio has shown to cause traffic spikes when trying to reconnect on bad connections.
  It *hosts* a socket.io server for its own clients and polls
  `/{org}/projects/{project}/deployment/target` over REST in `on_repeat` instead. `_DetectorState`
  (`_Initializing` / `_Updating` / `_ActiveDetector`) models the model swap: download to
  `models/<version>`, build a `DetectorLogic` through the factory, then swap atomically so the old
  model keeps serving unti