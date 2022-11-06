from enum import Enum
from learning_loop_node.context import Context
from learning_loop_node.trainer.downloader import TrainingsDownloader
from training import Training
from training import State as training_state


class SubState(str, Enum):
    Empty = "Empty"
    DataDownloaded = "DataDownloaded"


async def start(training: Training):
    if training.training_state != training_state.Preparing:
        raise Exception('Training is not in preparing state.')
    if training.training_sub_state is SubState.Empty:
        downloader = TrainingsDownloader(training.context))
        training.data=await downloader.download_training_data(training.images_folder)
        training.save()
