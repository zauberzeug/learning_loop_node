import uvicorn
from learning_loop_node.converter.converter_node import ConverterNode
from mock_converter import MockConverter
import backdoor_controls
import os
import logging

logging.basicConfig(level=logging.DEBUG)

mock_converter = MockConverter(source_format='mocked', target_format='mocked_converted')
node = ConverterNode(name='mocked converter', converter=mock_converter)
node.skip_check_state = True  # do not check states auotmatically for this mock

# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")


if __name__ == "__main__":
    reload_dirs = ['./restart'] if os.environ.get('MANUAL_RESTART', None) else ['./', './learning-loop-node']
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True, reload_dirs=reload_dirs)
