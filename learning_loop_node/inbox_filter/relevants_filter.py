from typing import List, Optional

from ..detector.outbox import Outbox
from ..detector.detections import Detections
from .relevants_group import RelevantsGroup


class RelevantsFilter():

    def __init__(self, outbox: Outbox) -> None:
        self.groups: dict = {}
        self.outbox: Outbox = outbox

    def learn(self, detections: Detections, mac: str, tags: Optional[str], raw_image: bytes) -> None:
        filter_causes = self._check_detections(detections, mac)
        if any(filter_causes):
            tags.append(mac)
            tags.append(*filter_causes)
            self.outbox.save(raw_image, detections, tags)

    def _check_detections(self, detections: Detections, mac: str) -> List[str]:

        {group.forget_old_detections() for (mac, group) in self.groups.items()}
        if mac not in self.groups:
            self.groups[mac] = RelevantsGroup()

        filter_causes = self.groups[mac].add_detections(detections)
        return filter_causes

    def reset(self) -> None:
        self.learners = {}
