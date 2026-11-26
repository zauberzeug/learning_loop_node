from unittest.mock import AsyncMock, Mock

import httpx

from ...data_classes import Context
from ...data_exchanger import DataExchanger
from ...loop_communication import LoopCommunicator
from ...trainer.downloader import TrainingsDownloader

CONTEXT = Context(organization='zauberzeug', project='demo')
IMAGE_IDS = ['3f2b8c5e-7a9d-4e0a-9d3f-6f1c7a524b1e', '9d106f1c-7a52-4b1e-8e0a-2b8c5e7a9d3f']


async def test_the_trainer_downloads_the_images_the_loop_chose_for_its_training(tmp_path):
    communicator = Mock(spec=LoopCommunicator)
    communicator.get = AsyncMock(return_value=httpx.Response(200, json={'image_ids': IMAGE_IDS}))
    data_exchanger = DataExchanger(CONTEXT, communicator)
    data_exchanger.download_images_data = AsyncMock(return_value=[])  # type: ignore[method-assign]
    data_exchanger.download_images = AsyncMock()  # type: ignore[method-assign]

    await TrainingsDownloader(data_exchanger, training_number=7).download_training_data(str(tmp_path))

    communicator.get.assert_awaited_once_with('/zauberzeug/projects/demo/trainings/7/image_ids')
    data_exchanger.download_images_data.assert_awaited_once_with(IMAGE_IDS)
