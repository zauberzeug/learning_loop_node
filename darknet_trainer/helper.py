from typing import List


def get_box_category_ids(data: dict) -> List[str]:
    return [c['id']for c in data['box_categories']]


def _get_box_category_names(data: dict) -> List[str]:
    return [c['name']for c in data['box_categories']]
