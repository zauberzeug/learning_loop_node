from learning_loop_node.trainer.downloader import Downloader
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.context import Context
from learning_loop_node.trainer.trainer import Trainer
from darknet_trainer import DarknetTrainer
from glob import glob
import os
from learning_loop_node.trainer.capability import Capability
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
from learning_loop_node import node


def get_files_from_data_folder():
    files = [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


def create_darknet_trainer() -> DarknetTrainer:
    return DarknetTrainer(uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4',
                          name='darknet trainer', capability=Capability.Box)


def create_downloader() -> Downloader:
    context = Context(organization='zauberzeug', project='pytest')
    return DownloaderFactory.create(server_base_url=node.SERVER_BASE_URL_DEFAULT, headers={}, context=context, capability=Capability.Box)


async def downlaod_data(trainer: Trainer):
    model_id = trainer_test_helper.assert_upload_model(
        ['darknet_tests/test_data/tiny_yolo.cfg', 'darknet_tests/test_data/fake_weightfile.weights'])
    context = Context(organization='zauberzeug', project='pytest')
    training = Trainer.generate_training(context, {'id': model_id})
    downloader = create_downloader()
    training.data = await downloader.download_data(training.images_folder, training.training_folder, model_id)
    trainer.training = training
