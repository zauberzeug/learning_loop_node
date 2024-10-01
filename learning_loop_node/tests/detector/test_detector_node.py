import os
import pytest
import numpy as np
from learning_loop_node.detector.detector_node import DetectorNode
from learning_loop_node.detector.detector_logic import DetectorLogic
from learning_loop_node.data_classes import Detections, BoxDetection

@pytest.fixture
def mock_detector_logic(monkeypatch):
    class MockDetectorLogic(DetectorLogic):
        def __init__(self):
            super().__init__('mock')

        @property
        def is_initialized(self):
            return True

        def evaluate_with_all_info(self, image, tags, source):
            return Detections(
                box_detections=[BoxDetection(category_name="test", 
                                             category_id="1",
                                             confidence=0.9, 
                                             x=0, y=0, width=10, height=10, 
                                             model_name="mock",
                                             )]
            )

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

    def mock_filtered_upload(*args, **kwargs):
        nonlocal filtered_upload_called
        filtered_upload_called = True

    def mock_save(*args, **kwargs):
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(detector_node.relevance_filter, "may_upload_detections", mock_filtered_upload)
    monkeypatch.setattr(detector_node.outbox, "save", mock_save)

    # Test cases
    test_cases = [
        (None, True, False),
        ("filtered", True, False),
        ("all", False, True),
        ("disabled", False, False),
    ]

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