from typing import Dict, List, Optional

from ...data_classes.image_metadata import ImageMetadata
from ..outbox import Outbox
from .cam_observation_history import CamObservationHistory


class RelevanceFilter():

    def __init__(self, outbox: Outbox) -> None:
        self.cam_histories: Dict[str, CamObservationHistory] = {}
        self.outbox: Outbox = outbox

    async def may_upload_detections(self,
                                    image_metadata: ImageMetadata,
                                    cam_id: str,
                                    raw_image: bytes,
                                    tags: List[str],
                                    source: Optional[str] = None,
                                    creation_date: Optional[str] = None) -> List[str]:
        """Check if the detection should be uploaded to the outbox.
        If so, upload it and return the list of causes for the upload.
        """
        for group in self.cam_histories.values():
            group.forget_old_detections()

        if cam_id not in self.cam_histories:
            self.cam_histories[cam_id] = CamObservationHistory()
        causes = self.cam_histories[cam_id].get_causes_to_upload(image_metadata)
        if len(image_metadata) >= 80:
            causes.append('unexpected_observations_count')
        if len(causes) > 0:
            tags = tags if tags is not None else []
            tags.extend(causes)
            await self.outbox.save(raw_image, image_metadata, tags, source, creation_date)
        return causes
