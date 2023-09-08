from typing import Dict, List

from ..data_classes.detections import Detections
from ..detector.outbox import Outbox
from .relevance_group import RelevanceGroup


class RelevanceFilter():

    def __init__(self, outbox: Outbox) -> None:
        self.groups: Dict[str, RelevanceGroup] = {}
        self.outbox: Outbox = outbox
        self.learners = {}

    def may_upload_detections(self, dets: Detections, cam_id: str, raw_image: bytes, tags: List[str]) -> List[str]:

        for group in self.groups.values():
            group.forget_old_detections()

        if cam_id not in self.groups:
            self.groups[cam_id] = RelevanceGroup()
        causes = self.groups[cam_id].add_detections(dets)
        if len(dets) >= 80:
            causes.append('unexpected_observations_count')
        if len(causes) > 0:
            tags = tags if tags is not None else []
            tags.extend(causes)
            self.outbox.save(raw_image, dets, tags)
        return causes

    def reset(self) -> None:
        self.learners = {}
