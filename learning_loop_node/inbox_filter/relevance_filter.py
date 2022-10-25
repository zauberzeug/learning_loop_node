from typing import List, Optional

from ..detector.outbox.outbox import Outbox
from ..detector.detections import Detections
from .relevance_group import RelevanceGroup


class RelevanceFilter():

    def __init__(self, outbox: Outbox) -> None:
        self.groups: dict[str, RelevanceGroup] = {}
        self.outbox: Outbox = outbox

    def learn(
        self,
        detections: Detections,
        camera_id: str,
        tags: Optional[str],
        raw_image: bytes
    ) -> List[str]:
        {group.forget_old_detections() for (_, group) in self.groups.items()}
        if camera_id not in self.groups:
            self.groups[camera_id] = RelevanceGroup()
        causes = self.groups[camera_id].add_detections(detections)
        if len(detections) >= 80:
            causes.append('unexpectedObservationsCount')
        if len(causes) > 0:
            tags.extend(causes)
            self.outbox.save(raw_image, detections, tags)
        return causes

    def reset(self) -> None:
        self.learners = {}
