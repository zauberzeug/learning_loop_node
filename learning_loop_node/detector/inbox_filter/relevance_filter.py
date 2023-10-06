from typing import Dict, List

from ...data_classes.detections import Detections
from ..outbox import Outbox
from .cam_observation_history import CamObservationHistory


class RelevanceFilter():

    def __init__(self, outbox: Outbox) -> None:
        self.cam_histories: Dict[str, CamObservationHistory] = {}
        self.outbox: Outbox = outbox

    def may_upload_detections(self, dets: Detections, cam_id: str, raw_image: bytes, tags: List[str]) -> List[str]:
        for group in self.cam_histories.values():
            group.forget_old_detections()

        if cam_id not in self.cam_histories:
            self.cam_histories[cam_id] = CamObservationHistory()
        causes = self.cam_histories[cam_id].get_causes_to_upload(dets)
        if len(dets) >= 80:
            causes.append('unexpected_observations_count')
        if len(causes) > 0:
            tags = tags if tags is not None else []
            tags.extend(causes)
            self.outbox.save(raw_image, dets, tags)
        return causes
