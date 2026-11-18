"""The loop names the states (such as "complete" or "review") a training takes each set's images from: `{set: [state]}`.

A set that is not named takes its images from "complete" only, which is what every training did before the loop
could name states. An image is used when its own state is one of the states of its set. This runs before the
download and before `assign_validation_set`, so a computed validation set comes from the kept train images.
"""

import logging
from collections.abc import Collection, Mapping

logger = logging.getLogger(__name__)

SETS = ('train', 'val', 'test')
DEFAULT_STATES = ('complete',)


def states_to_fetch(states: Mapping[str, Collection[str]]) -> list[str]:
    """Every state at least one set takes images from, each named once."""
    return list(dict.fromkeys(state for image_set in SETS for state in states.get(image_set) or DEFAULT_STATES))


def restrict_to_states(image_data: list[dict], states: Mapping[str, Collection[str]]) -> list[dict]:
    """Keep the images whose state is one of the states of their set."""
    if not states:
        return image_data
    kept = [image for image in image_data if image.get('state') in (states.get(image['set']) or DEFAULT_STATES)]
    logger.info('Restricted the training to the states %s: kept %d of %d images',
                dict(states), len(kept), len(image_data))
    return kept
