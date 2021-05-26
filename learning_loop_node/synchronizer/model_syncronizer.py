from typing import List
from learning_loop_node.context import Context
from pydantic.main import BaseModel
from learning_loop_node import node_helper


class ModelSynchronizer(BaseModel):
    server_base_url: str
    headers: dict
    context: Context

    def download(self, target_folder: str, model_id: str) -> List[str]:
        node_helper.download_model(self.server_base_url, self.headers, target_folder, self.context.organization,
                                   self.context.project, model_id)

    async def upload(self, files: List[str], model_id: str) -> None:
        await node_helper.upload_model(self.server_base_url, self.headers, self.context.organization,
                                       self.context.project, files,  model_id)
