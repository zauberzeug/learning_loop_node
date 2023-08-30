from typing import Dict, List

from ..detector.detections import Detections
from ..detector.outbox.outbox import Outbox
from .relevance_group import RelevanceGroup


class RelevanceFilter():

    def __init__(self, outbox: Outbox) -> None:
        self.groups: Dict[str, RelevanceGroup] = {}
        self.outbox: Outbox = outbox
        self.learners = {}

    def learn(self, detections: Detections, camera_id: str, raw_image: bytes, tags: List[str]) -> List[str]:
        _ = {group.forget_old_detections() for (_, group) in self.groups.items()}  # TODO: what?
        if camera_id not in self.groups:
            self.groups[camera_id] = RelevanceGroup()
        causes = self.groups[camera_id].add_detections(detections)
        if len(detections) >= 80:
            causes.append('unexpectedObservationsCount')
        if len(causes) > 0:
            tags = tags if tags is not None else []
            tags.extend(causes)
            self.outbox.save(raw_image, detections, tags)
        return causes

    def reset(self) -> None:
        self.learners = {}
