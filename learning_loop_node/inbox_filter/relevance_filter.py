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
        causes = self.groups[camera_id].add_detections(detections, criteria=submission_criteria)
        if len(detections) >= 80:
            causes.append('unexpectedObservationsCount')
        if len(causes) > 0 or submission_criteria == '':  # NOTE if no criteria are defined, all images are taken
            if len(causes) > 0:
                tags.append(*causes)
            tags.append(camera_id)
            self.outbox.save(raw_image, detections, tags)
        return causes

    def reset(self) -> None:
        self.learners = {}
