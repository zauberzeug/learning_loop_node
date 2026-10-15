"""Delete the projects CI runs left behind on the shared loop.

Every suite deletes its project in its teardown, but a job that is cancelled or loses its runner never gets
there. CI runs this as the last step of every job, `if: always()`, so it also runs after a cancellation:

    python -m learning_loop_node.tests.cleanup_test_projects

The loop keeps no creation time for a project, so the run named in its suffix decides whether it is stale --
it is deleted once that run is completed, or has been re-run since. A run still in progress keeps its
projects, and so does a run this repository does not know.
"""
import asyncio
import logging
import os
import re
from typing import Optional

import httpx

from ..loop_communication import LoopCommunicator
from .test_helper import TEST_PROJECT_SUFFIX

ORGANIZATION = 'zauberzeug'
# the projects the suites create and delete -- only these are ever swept, so a project of another repository's
# CI that happens to follow the same naming is left alone
PROJECTS = ('pytest_nodelib_general', 'pytest_nodelib_annotator', 'pytest_mock_trainer_test1', 'pytest_mock_trainer_test2')
SUFFIXED = re.compile(rf'^(?:{"|".join(PROJECTS)})_(?P<run_id>\d+)_(?P<attempt>\d+)_\d+$')


async def run_status(client: httpx.AsyncClient, run_id: str) -> Optional[dict]:
    response = await client.get(f'/repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run_id}')
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def is_stale(project: str, run: Optional[dict]) -> bool:
    match = SUFFIXED.match(project)
    assert match is not None
    if project.endswith(f'_{os.environ[TEST_PROJECT_SUFFIX]}'):
        return True  # this job's own, and its tests are over
    if run is None:
        return False
    return run['status'] == 'completed' or int(match['attempt']) < run['run_attempt']


async def main() -> None:
    loop_communicator = LoopCommunicator()
    github = httpx.AsyncClient(base_url='https://api.github.com', timeout=30,
                               headers={'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}',
                                        'Accept': 'application/vnd.github+json'})
    try:
        response = await loop_communicator.get('/projects')
        response.raise_for_status()
        listed = {p['project_id'] for p in response.json()['projects']
                  if p['organization_id'] == ORGANIZATION and SUFFIXED.match(p['project_id'])}
        logging.info('%d test projects listed', len(listed))
        # this job's own are deleted by name too, in case the listing does not show them
        own = {f'{project}_{os.environ[TEST_PROJECT_SUFFIX]}' for project in PROJECTS}
        runs: dict = {}
        for project in sorted(listed | own):
            run_id = SUFFIXED.match(project)['run_id']  # type: ignore[index]
            if run_id not in runs:
                runs[run_id] = await run_status(github, run_id)
            if not is_stale(project, runs[run_id]):
                logging.info('keeping %s', project)
                continue
            response = await loop_communicator.delete(f'/{ORGANIZATION}/projects/{project}', timeout=30)
            logging.info('deleting %s: %s', project, response.status_code)
    finally:
        await github.aclose()
        await loop_communicator.shutdown()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    asyncio.run(main())
