
import os

import uvicorn
from app_code.mock_annotator import MockAnnotatorLogic

from learning_loop_node.annotation.annotator_node import AnnotatorNode

tool = MockAnnotatorLogic()
node = AnnotatorNode(uuid='00000000-1111-2222-3333-444444444444',
                     name=f'Annotation Node {os.uname()[1]}', annotator_logic=tool)


if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', use_colors=True, reload=True)
