"""The loop restricts sets to subsets (named groups of images such as "winter") with `{set: [{uuid, name}]}`.

A restricted set keeps the images in any of its subsets, an unrestricted set all its images. The images are matched
by subset uuid, so a subset renamed in the loop meanwhile still matches. This runs before the download and before
`assign_validation_set`, so a computed validation set comes from the restricted train images.
"""

import logging
from collections.abc import Collection, Mapping

logger = logging.getLogger(__name__)


def restrict_to_subsets(image_data: list[dict], subsets: Mapping[str, Collection[Mapping[str, str]]]) -> list[dict]:
    if not subsets:
        return image_data
    chosen = {image_set: {entry['uuid'] for entry in entries} for image_set, entries in subsets.items()}
    kept = [image for image in image_data
            if image['set'] not in chosen or not _subset_ids(image).isdisjoint(chosen[image['set']])]
    logger.info('Restricted the training to the subsets %s: kept %d of %d images',
                subset_names(subsets), len(kept), len(image_data))
    return kept


def subset_names(subsets: Mapping[str, Collection[Mapping[str, str]]]) -> dict[str, list[str]]:
    return {image_set: [entry['name'] for entry in entries] for image_set, entries in subsets.items()}


def _subset_ids(image: dict) -> set[str]:
    return {subset['id'] for subset in image.get('subsets') or []}
