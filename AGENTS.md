# AI Agent Guidelines for the Learning Loop Node Library

`learning_loop_node` is the **public Python library** every node uses to talk to the
[Learning Loop](https://learning-loop.ai). Four node types build on it: Trainer, Detector,
Annotator and Converter. It is published to PyPI, so its public surface is an API that
`loop`, `dfine_node`, `yolov5_node` and `classification_node` depend on.

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

## Running and testing

The suites talk to a real Learning Loop instance and read their credentials from a local `.env`
(`LOOP_HOST`, `LOOP_USERNAME`, `LOOP_PASSWORD`). Without a reachable loop they cannot pass — do not
treat their failure as a regression you introduced.

```bash
./run_tests.sh              # all suites
./run_tests.sh <filter>     # passed to pytest as -k
```

There is no `.pre-commit-config.yaml` here and no ruff in the project environment, despite what the
shared Linting section says. Lint with:

```bash
uvx ruff check .
```

A clean tree already reports several hundred ruff findings, so a clean run is not a reachable goal.
Compare the count on the files you touched, before and after.

`.github/workflows/pytest.yml` runs the suites, `publish.yml` releases to PyPI on a tagged release.

## Working in this repository

- **This is a library — declaring a dependency is part of its API.** Before removing or loosening
  one, grep the consumers (`../loop`, `../dfine_node`, `../yolov5_node`,
  `../classification_node`) for the package: a consumer that imports it without declaring it
  inherits it from here and breaks when it goes away.
- **Renaming or reshaping anything exported** breaks those four repositories. Say so in the pull
  request and check whether a companion change is needed there.
- `../loop` checks this repository out as its `nodes` symlink, so a local change is visible to a
  local loop immediately — but only a released version reaches CI and production.
- Bump `version` in `pyproject.toml` for a release; the trainer nodes pin the library version in
  their image tags (`A.B.C-nlvX.Y.Z`).
