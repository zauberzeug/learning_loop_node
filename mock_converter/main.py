from learning_loop_node.converter.mock_converter import MockConverter
import uvicorn
from learning_loop_node.converter.converter_node import ConverterNode
import backdoor_controls

mock_converter = MockConverter()
converter_node = ConverterNode(uuid='85ef1a58-308d-4c80-8931-43d1f752f4f3',
                               name='mocked converter', converter=mock_converter)

# setting up backdoor_controls
converter_node.include_router(backdoor_controls.router, prefix="")


if __name__ == "__main__":
    uvicorn.run("main:converter_node", host="0.0.0.0", port=80, lifespan='on', reload=True, reload_dirs=['./restart'])
