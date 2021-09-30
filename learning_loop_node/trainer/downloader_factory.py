from learning_loop_node.context import Context
from learning_loop_node.trainer.downloader import DataDownloader


class DownloaderFactory:

    @staticmethod
    def create(context: Context) -> DataDownloader:
        return DataDownloader(
            context=context,
            data_query_params='state=complete'
        )
