
import os

import uvicorn
from mock_annotation_tool import MockAnnotatorNode

from learning_loop_node.annotation.annotator_node import AnnotatorNode
from mock_annotation_node import backdoor_controls

tool = MockAnnotatorNode()
node = AnnotatorNode(uuid='00000000-1111-2222-3333-444444444444',
                     name=f'Annotation Node {os.uname()[1]}', annotator_logic=tool)

node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', use_colors=True, reload=True)
