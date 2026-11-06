import json
from collections.abc import Coroutine
from dataclasses import asdict
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pytest

from learning_loop_node.data_classes import BatchImageMetadata, ImageMetadata, ModelInformation
from learning_loop_node.detector import detector_node as detector_node_module
from learning_loop_node.detector.detector_node import DetectorNode, _batch_image_metadata
from learning_loop_node.helpers.misc import ImageData

from ..test_helper import DETECTOR_PROJECT, project_name
from .conftest import MockDetectorLogic

PROJECT = project_name(DETECTOR_PROJECT)


@pytest.mark.asyncio
async def test_get_detections(detector_node: DetectorNode, monkeypatch):
    # Mock raw image data
    np_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock relevance_filter and outbox
    filtered_upload_called = False
    save_called = False

    save_args = []

    async def mock_filtered_upload(*args, **kwargs) -> List[str]:  # pylint: disable=unused-argument
        nonlocal filtered_upload_called
        filtered_upload_called = True
        return []

    async def mock_save(*args, **kwargs):
        nonlocal save_called
        nonlocal save_args
        save_called = True
        save_args = (args, kwargs)

    monkeypatch.setattr(detector_node.relevance_filter, "may_upload_detections", mock_filtered_upload)
    monkeypatch.setattr(detector_node.outbox, "save", mock_save)

    created_tasks = []

    def mock_create_task(coroutine, *args, **kwargs):
        created_tasks.append(coroutine)
    monkeypatch.setattr(detector_node_module.background_tasks, "create", mock_create_task)

    # Test cases
    test_cases: List[Tuple[Literal['filtered', 'all', 'disabled'], bool, bool]] = [
        ("filtered", True, False),
        ("all", False, True),
        ("disabled", False, False),
    ]

    expected_save_args = {
        'image': np_image,
        'detections': detector_node._detector.logic.image_metadata,  # type: ignore
    }

    for autoupload, expect_filtered, expect_all in test_cases:
        filtered_upload_called = False
        save_called = False
        created_tasks.clear()

        result = await detector_node.get_detections(
            image=np_image,
            camera_id="test_camera",
            tags=["test_tag"],
            source="test_source",
            creation_date="2024-01-01T00:00:00",
            autoupload=autoupload
        )

        for task in created_tasks:
            await task

        # Check if detections were processed
        assert result is not None
        assert result.box_detections is not None
        assert len(result.box_detections) == 1
        assert result.box_detections[0].category_name == "test"

        # Check if the correct upload method was called
        assert filtered_upload_called == expect_filtered
        assert save_called == expect_all

        if save_called:
            save_pos_args, save_kwargs = save_args  # pylint: disable=unbalanced-tuple-unpacking
            expected_values = list(expected_save_args.values())
            assert len(save_pos_args) + len(save_kwargs) == len(expected_values)

            # Check positional arguments
            for arg, expected in zip(save_pos_args, expected_values[:len(save_pos_args)]):
                if isinstance(arg, (list, np.ndarray)):
                    assert np.array_equal(arg, expected)
                else:
                    assert arg == expected

            # Check keyword arguments
            for key, value in save_kwargs.items():
                expected = expected_save_args[key]
                if isinstance(value, (list, np.ndarray)):
                    assert np.array_equal(value, expected)
                else:
                    assert value == expected


@pytest.mark.parametrize('node_conf_threshold, expected', [(0.35, 0.35), (None, 0.6)])
async def test_the_factory_gets_the_conf_threshold_of_the_node_else_the_models(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_conf_threshold: float | None, expected: float):
    monkeypatch.setenv('LOOP_ORGANIZATION', 'zauberzeug')
    monkeypatch.setenv('LOOP_PROJECT', PROJECT)
    model_info = ModelInformation(id='test', host='', organization='zauberzeug', project=PROJECT, version='0.0',
                                  conf_threshold=0.6)
    (tmp_path / 'model.json').write_text(json.dumps(asdict(model_info)))
    factory = _RecordingFactory()
    node = DetectorNode(name='test_node', detector_factory=factory, conf_threshold=node_conf_threshold)

    await node._build_and_swap_detector(str(tmp_path))  # pylint: disable=protected-access

    assert factory.model_info is not None
    assert factory.model_info.conf_threshold == expected
    assert json.loads((tmp_path / 'model.json').read_text())['conf_threshold'] == 0.6


class _RecordingFactory:
    model_format = 'mock'

    def __init__(self) -> None:
        self.model_info: ModelInformation | None = None

    async def build(self, model_info: ModelInformation) -> MockDetectorLogic:
        self.model_info = model_info
        return MockDetectorLogic(model_info)


async def test_batch_images_carry_their_own_camera_source_and_tags(detector_node: DetectorNode,
                                                                   filter_calls: '_FilterCalls'):
    result = await detector_node.get_batch_detections(
        images=_images(3),
        tags=['robot'],
        camera_id='batch_cam',
        source='robot',
        creation_date='2024-01-01T00:00:00',
        autoupload='filtered',
        image_metadata=[BatchImageMetadata(camera_id='cam_a', source='robot/cam_a', tags=['cam_a']),
                        BatchImageMetadata(camera_id='cam_b', source='robot/cam_b', creation_date='2024-02-02T00:00:00'),
                        BatchImageMetadata()],
    )
    await filter_calls.settle()

    assert filter_calls.calls == [('cam_a', 'robot/cam_a', ['robot', 'cam_a']),
                                  ('cam_b', 'robot/cam_b', ['robot']),
                                  ('batch_cam', 'robot', ['robot'])], 'what an image does not set comes from the batch'
    assert [item.created for item in result.items] == \
        ['2024-01-01T00:00:00', '2024-02-02T00:00:00', '2024-01-01T00:00:00']


async def test_batch_without_image_metadata_applies_the_batch_values_to_every_image(detector_node: DetectorNode,
                                                                                    filter_calls: '_FilterCalls'):
    await detector_node.get_batch_detections(images=_images(2), tags=['robot'], camera_id='batch_cam', source='robot')
    await filter_calls.settle()

    assert filter_calls.calls == [('batch_cam', 'robot', ['robot'])] * 2


async def test_sio_batch_detect_reads_the_values_an_image_dict_sets(detector_node: DetectorNode,
                                                                    filter_calls: '_FilterCalls'):
    batch_detect = detector_node.sio.handlers['/']['batch_detect']
    image = {'bytes': _images(1)[0].tobytes(order='C'), 'dtype': 'uint8', 'shape': (10, 10, 3)}

    result = await batch_detect('sid', {'images': [{**image, 'camera_id': 'cam_a', 'tags': ['cam_a']}, image],
                                        'camera_id': 'batch_cam', 'tags': ['robot'], 'source': 'robot'})
    await filter_calls.settle()

    assert len(result['items']) == 2
    assert filter_calls.calls == [('cam_a', 'robot', ['robot', 'cam_a']), ('batch_cam', 'robot', ['robot'])]


async def test_batch_rejects_image_metadata_that_does_not_match_the_images(detector_node: DetectorNode):
    with pytest.raises(ValueError):
        await detector_node.get_batch_detections(images=_images(2), tags=[], image_metadata=[BatchImageMetadata()])


@pytest.mark.parametrize('image_data', [{}, {'tags': None}, {'tags': []}])
def test_an_image_without_tags_adds_none(image_data: dict):
    assert _batch_image_metadata(image_data) == BatchImageMetadata()


@pytest.mark.parametrize('tags', ['cam_a', ('cam_a',), 7])
def test_an_image_with_tags_that_are_no_list_is_rejected(tags: object):
    with pytest.raises(ValueError):
        _batch_image_metadata({'tags': tags})


class _FilterCalls:
    """Records the camera id, source and tags of every image the relevance filter is asked about."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None, list[str]]] = []
        self._tasks: list[Coroutine] = []

    def create_task(self, coroutine: Coroutine, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        self._tasks.append(coroutine)

    async def may_upload_detections(self, image_metadata: ImageMetadata, cam_id: str | None,
                                    image: ImageData) -> list[str]:  # pylint: disable=unused-argument
        self.calls.append((cam_id, image_metadata.source, list(image_metadata.tags)))
        return []

    async def settle(self) -> None:
        """Run the background tasks the node created."""
        for task in self._tasks:
            await task
        self._tasks.clear()


@pytest.fixture
def filter_calls(detector_node: DetectorNode, monkeypatch: pytest.MonkeyPatch) -> _FilterCalls:
    recorder = _FilterCalls()
    monkeypatch.setattr(detector_node.relevance_filter, 'may_upload_detections', recorder.may_upload_detections)
    monkeypatch.setattr(detector_node_module.background_tasks, 'create', recorder.create_task)
    return recorder


def _images(count: int) -> list[np.ndarray]:
    return [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(count)]
