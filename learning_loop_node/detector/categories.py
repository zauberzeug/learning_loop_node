"""Resolving a model's categories from what it emits.

A model reports class indices, or names, and only ``ModelInformation.categories`` gives those
meaning. Both lookups are the point at which a model and its metadata are checked against each
other, so both raise rather than skipping quietly: a mismatch here makes every prediction on the
image suspect, and it is far cheaper to diagnose at the lookup than three layers later.
"""

from ..data_classes import Category, ModelInformation


def category_by_index(model_information: ModelInformation, index: int) -> Category:
    """Resolve the category a model's class index refers to.

    Models emit class indices, and the order of ``model_information.categories`` is what
    gives them meaning — so an out-of-range index is a model/metadata mismatch, not a
    detection to skip quietly.

    :raises ValueError: If the index is outside the model's category list.
    """
    categories = model_information.categories
    if not 0 <= index < len(categories):
        raise ValueError(
            f'category index {index} is out of range for a model with {len(categories)} categories')
    return categories[index]


def category_by_name(model_information: ModelInformation, name: str) -> Category:
    """Resolve a category by name, for models whose outputs are named rather than indexed.

    :raises ValueError: If no category of that name exists.
    """
    for category in model_information.categories:
        if category.name == name:
            return category
    known = ', '.join(category.name for category in model_information.categories)
    raise ValueError(f'unknown category name {name!r}; the model knows: {known}')
