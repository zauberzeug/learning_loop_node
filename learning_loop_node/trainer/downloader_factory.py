from learning_loop_node.context import Context
from learning_loop_node.trainer.capability import Capability
from learning_loop_node.trainer.downloader import Downloader


class DownloaderFactory:

    @staticmethod
    def create(server_base_url: str, headers: dict, context: Context, capability: Capability):
        if capability == Capability.Box:
            return Downloader(
                server_base_url=server_base_url,
                headers=headers,
                context=context,
                data_query_params='state=complete&mode=box'
            )
        else:
            raise NotImplementedError('Not implemented yet.')
