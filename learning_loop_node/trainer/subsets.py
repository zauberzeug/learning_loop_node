"""The loop restricts sets to subsets (named groups of images such as "winter") with `{set: [subset names]}`.

A restricted set keeps the images in any of its subsets, an unrestricted set all its images. This runs before the
download and before `assign_validation_set`, so a computed validation set comes from the restricted train images.
"""

import logging
from collections.abc import Collection, Mapping

logger = logging.getLogger(__name__)


def restrict_to_subsets(image_data: list[dict], subsets: Mapping[str, Collection[str]]) -> list[dict]:
    if not subsets:
        return image_data
    kept = [image for image in image_data
            if image['set'] not in subsets or not _names(image).isdisjoint(subsets[image['set']])]
    logger.info('Restricted the training to the subsets %s: kept %d of %d images', subsets, len(kept), len(image_data))
    return kept


def _names(image: dict) -> set[str]:
    return {subset['name'] for subset in image.get('subsets') or []}
