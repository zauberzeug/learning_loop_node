"""Delete the projects CI runs left behind on the shared loop.

Every suite deletes its project in its teardown, but a job that is cancelled or loses its runner never gets
there. CI runs this as the last step of every job, `if: always()`, so it also runs after a cancellation:

    python -m learning_loop_node.tests.cleanup_test_projects

The loop keeps no creation time for a project, so the run named in its suffix decides whether it is stale --
it is deleted once that run is completed, or has been re-run since. A run still in progress keeps its
projects, and so does a run this repository does not know or that cannot be looked up. The job's own projects go
first; a delete that times out ends the sweep, because the loop is still deleting and the next one would block.
"""
import asyncio
import logging
import os
import re

import httpx

from ..loop_communication import LoopCommunicator
from .test_helper import TEST_PROJECT_SUFFIX, TEST_PROJECTS

logger = logging.getLogger('cleanup_test_projects')

ORGANIZATION = 'zauberzeug'
# only the suites' own projects are ever swept, so a project of another repository's CI that happens to follow the
# same naming is left alone
SUFFIXED = re.compile(rf'^(?:{"|".join(TEST_PROJECTS)})_(?P<run_id>\d+)_(?P<attempt>\d+)_\d+$')


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
        logger.info('%d test projects listed', len(listed))
        # this job's own come first and go by name, in case the listing does not show them; its tests are over, so
        # they need no lookup and a failing one cannot keep them
        own = [f'{project}_{os.environ[TEST_PROJECT_SUFFIX]}' for project in TEST_PROJECTS]
        for project in own + await stale_projects(github, listed - set(own)):
            try:
                response = await loop_communicator.delete(f'/{ORGANIZATION}/projects/{project}', timeout=75)
            except httpx.TimeoutException:
                # the loop goes on deleting after the client gave up, and another delete now would only block on it
                logger.warning('deleting %s timed out, the remaining projects are left to the next sweep', project)
                break
            logger.info('deleting %s: %s %s', project, response.status_code,
                         response.text if response.status_code not in (200, 404) else '')
    finally:
        await github.aclose()
        await loop_communicator.shutdown()


async def stale_projects(client: httpx.AsyncClient, projects: set[str]) -> list[str]:
    runs: dict[str, dict | None] = {}
    stale = []
    for project in sorted(projects):
        run_id = SUFFIXED.match(project)['run_id']  # type: ignore[index]
        if run_id not in runs:
            try:
                runs[run_id] = await run_status(client, run_id)
            except httpx.HTTPError:
                logger.exception('looking up run %s failed, keeping its projects', run_id)
                runs[run_id] = None
        if is_stale(project, runs[run_id]):
            stale.append(project)
        else:
            logger.info('keeping %s', project)
    return stale


async def run_status(client: httpx.AsyncClient, run_id: str) -> dict | None:
    response = await client.get(f'/repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run_id}')
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def is_stale(project: str, run: dict | None) -> bool:
    match = SUFFIXED.match(project)
    assert match is not None
    if run is None:
        return False
    return run['status'] == 'completed' or int(match['attempt']) < run['run_attempt']


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    asyncio.run(main())
