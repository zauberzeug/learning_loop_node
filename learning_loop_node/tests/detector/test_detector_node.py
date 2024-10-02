import os
import pytest
from numpy.typing import ArrayLike
import numpy as np
from learning_loop_node.detector.detector_node import DetectorNode
from learning_loop_node.detector.detector_logic import DetectorLogic
from learning_loop_node.data_classes import Detections, BoxDetection

@pytest.fixture
def mock_detector_logic(monkeypatch):
    class MockDetectorLogic(DetectorLogic):
        def __init__(self):
            super().__init__('mock')
            self.detections = Detections(
                box_detections=[BoxDetection(category_name="test", 
                                             category_id="1",
                                             confidence=0.9, 
                                             x=0, y=0, width=10, height=10, 
                                             model_name="mock",
                                             )]
            )

        @property
        def is_initialized(self):
            return True

        def evaluate_with_all_info(self, image, tags, source):
            return self.detections

    return MockDetectorLogic()

@pytest.fixture
def detector_node(mock_detector_logic):
    os.environ['ORGANIZATION'] = 'test_organization'
    os.environ['PROJECT'] = 'test_project'
    return DetectorNode(name="test_node", detector=mock_detector_logic)

@pytest.mark.asyncio
async def test_get_detections(detector_node: DetectorNode, monkeypatch):
    # Mock raw image data
    raw_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock relevance_filter and outbox
    filtered_upload_called = False
    save_called = False

    save_args = []

    def mock_filtered_upload(*args, **kwargs):
        nonlocal filtered_upload_called
        filtered_upload_called = True

    def mock_save(*args, **kwargs):
        nonlocal save_called
        nonlocal save_args
        save_called = True
        save_args = (args, kwargs)

    monkeypatch.setattr(detector_node.relevance_filter, "may_upload_detections", mock_filtered_upload)
    monkeypatch.setattr(detector_node.outbox, "save", mock_save)

    # Test cases
    test_cases = [
        (None, True, False),
        ("filtered", True, False),
        ("all", False, True),
        ("disabled", False, False),
    ]

    expected_save_args = {
        'image': raw_image,
        'detections': detector_node.detector_logic.detections, # type: ignore
        'tags': ['test_tag'],
        'source': 'test_source',
    }

    for autoupload, expect_filtered, expect_all in test_cases:
        filtered_upload_called = False
        save_called = False

        result = await detector_node.get_detections(
            raw_image=raw_image,
            camera_id="test_camera",
            tags=["test_tag"],
            source="test_source",
            autoupload=autoupload
        )

        # Check if detections were processed
        assert result is not None
        assert "box_detections" in result
        assert len(result["box_detections"]) == 1
        assert result["box_detections"][0]["category_name"] == "test"

        # Check if the correct upload method was called
        assert filtered_upload_called == expect_filtered
        assert save_called == expect_all

        if save_called:
            save_pos_args, save_kwargs = save_args
            expected_values = list(expected_save_args.values())
            
            # Check positional arguments
            for arg, expected in zip(save_pos_args, expected_values[:len(save_pos_args)]):
                if isinstance(arg, list) or isinstance(arg, np.ndarray):
                    assert np.array_equal(arg, expected)
                else:
                    assert arg == expected
            
            # Check keyword arguments
            for key, value in save_kwargs.items():
                expected = expected_save_args[key]
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    assert np.array_equal(value, expected)
                else:
                    assert value == expected
