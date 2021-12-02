from typing import List, Tuple


def create_resource_paths(organization_name: str, project_name: str, image_ids: List[str]) -> Tuple[List[str], List[str]]:
    if not image_ids:
        return [], []
    url_ids = [(f'api/{organization_name}/projects/{project_name}/images/{id}/main', id)
               for id in image_ids]
    urls, ids = list(map(list, zip(*url_ids)))

    return urls, ids
