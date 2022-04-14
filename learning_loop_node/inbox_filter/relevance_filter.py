from typing import List, Optional

from ..detector.outbox import Outbox
from ..detector.detections import Detections
from .relevance_group import RelevanceGroup

DEFAULT_SUBMISSION_CRITERIA = 'novel,uncertain'


class RelevanceFilter():

    def __init__(self, outbox: Outbox) -> None:
        self.groups: dict[str, RelevanceGroup] = {}
        self.outbox: Outbox = outbox

    def learn(
        self,
        detections: Detections,
        camera_id: str,
        tags: Optional[str],
        raw_image: bytes,
        submission_criteria='unsure'
    ) -> List[str]:
        {group.forget_old_detections() for (_, group) in self.groups.items()}
        if camera_id not in self.groups:
            self.groups[camera_id] = RelevanceGroup()
        filter_causes = self.groups[camera_id].add_detections(detections, criteria=submission_criteria)
        if len(detections) >= 80:
            filter_causes.append('unexpectedObservationsCount')
        if any(filter_causes):
            tags.append(camera_id)
            tags.append(*filter_causes)
            self.outbox.save(raw_image, detections, tags)
        return filter_causes

    def reset(self) -> None:
        self.learners = {}
