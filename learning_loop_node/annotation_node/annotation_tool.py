from abc import abstractmethod
from pydantic import BaseModel
from learning_loop_node.annotation_node.data_classes import UserInput,ToolOutput 


class AnnotationTool(BaseModel):

    @abstractmethod
    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:
        pass


class EmptyAnnotationTool():
    async def handle_user_input(self, user_input: UserInput) -> ToolOutput:
        return ToolOutput(svg="", shape={})

