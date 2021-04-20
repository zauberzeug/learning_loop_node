#!/usr/bin/env python3
import argparse
import asyncio
from asyncio.queues import QueueEmpty
import base64
import aiohttp
from icecream import ic
from asyncio import Queue
from typing import List
import aiofiles
import requests
import node_helper
import time


async def download_ids(base_url, organization_name, project_name, headers) -> dict:
    ids_url = f'{base_url}/api/{organization_name}/projects/{project_name}/data/data2'
    async with aiohttp.ClientSession() as session:
        ic()
        async with session.get(ids_url, headers=headers) as response:
            ic()
            assert response.status == 200, f'Error during downloading list of ids. Statuscode is {response.status}'
            data = await response.json()

    return data


async def main(loop: asyncio.BaseEventLoop, host_base_url, organization_name, project_name, headers):
    t = time.time()
    ids = (await download_ids(host_base_url, organization_name, project_name, headers))['image_ids']
    ic(f'downloaded {len(ids)} ids. Time : {time.time() - t}')

    image_data_coroutine = node_helper.download_images_data(
        host_base_url, headers, organization_name, project_name, ids, 100)
    image_data_task = loop.create_task(image_data_coroutine)

    # urls, ids = node_helper.create_resource_urls(
    #     host_base_url, organization_name, project_name, ids)
    # await node_helper.download_images(loop, urls, ids, headers, "/data")

    image_data = await image_data_task
    ic(f'Done downloading image_data for {len(image_data)} images.')

    ic(f'Done downloading image_data for {len(image_data)} images.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import a project from the old LL.')
    parser.add_argument("organization_name", nargs=1, help='the name of the organization')
    parser.add_argument("project_name", nargs=1, help='the name of the project')
    parser.add_argument("host_base_url", nargs='?', default='http://localhost', type=str,
                        help='the base url of the host. default is "http://localhost"')
    parser.add_argument("username", nargs='?', default=None, type=str, help='the username for the basic auth.')
    parser.add_argument("password", nargs='?', default=None, type=str, help='the password for the basic auth.')
    args = parser.parse_args()

    organization_name = args.organization_name[0]
    project_name = args.project_name[0]
    host_base_url = args.host_base_url

    headers = {}
    if args.username and args.username:
        username = args.username
        password = args.password
        headers["Authorization"] = "Basic " + \
            base64.b64encode(f"{username}:{password}".encode()).decode()

    loop = asyncio.get_event_loop()
    main_task = loop.create_task(main(loop, host_base_url, organization_name, project_name, headers))
    loop.run_until_complete(main_task)
