"""Delete the test projects CI jobs left behind on the shared loop.

CI runs this as the last step of every job, also after a cancellation:

    python -m learning_loop_node.tests.cleanup_test_projects

The job's own projects go first. Any other is deleted once the run named in its suffix is completed or has been
re-run since; a run in progress or one that cannot be looked up keeps its projects. A delete that times out ends the
sweep, since the loop goes on deleting and the next delete would block on it.
"""
import asyncio
import logging
import os
import re

import httpx

from ..loop_communication import LoopCommunicator
from .test_helper import DELETE_TIMEOUT, TEST_PROJECT_SUFFIX, TEST_PROJECTS

logger = logging.getLogger('learning_loop_node.tests.cleanup_test_projects')

ORGANIZATION = 'zauberzeug'
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
        # by name, in case the listing misses them
        own = [f'{project}_{os.environ[TEST_PROJECT_SUFFIX]}' for project in TEST_PROJECTS]
        for project in own + await stale_projects(github, listed - set(own)):
            try:
                response = await loop_communicator.delete(f'/{ORGANIZATION}/projects/{project}',
                                                          timeout=DELETE_TIMEOUT)
            except httpx.TimeoutException:
                logger.warning('deleting %s timed out, the remaining projects are left to the next sweep', project)
                break
            logger.info('deleting %s: %s %s', project, response.status_code,
                         response.text if response.status_code not in (200, 404) else '')
    finally:
        await github.aclose()
        await loop_communicator.shutdown()


async def stale_projects(client: httpx.AsyncClient, projects: set[str]) -> list[str]:
    by_run: dict[str, list[re.Match]] = {}
    for project in sorted(projects):
        match = SUFFIXED.match(project)
        assert match is not None
        by_run.setdefault(match['run_id'], []).append(match)
    stale = []
    for run_id, matches in by_run.items():
        try:
            run = await run_status(client, run_id)
        except httpx.HTTPError as e:
            logger.warning('keeping the projects of run %s, it cannot be looked up: %s', run_id, e)
            continue
        for match in matches:
            if run['status'] == 'completed' or int(match['attempt']) < run['run_attempt']:
                stale.append(match.string)
            else:
                logger.info('keeping %s', match.string)
    return stale


async def run_status(client: httpx.AsyncClient, run_id: str) -> dict:
    response = await client.get(f'/repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run_id}')
    response.raise_for_status()
    return response.json()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    asyncio.run(main())
