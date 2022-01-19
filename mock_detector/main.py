import logging
from learning_loop_node import DetectorNode, ModelInformation
from learning_loop_node.data_classes import Category
import backdoor_controls
from mock_detector import MockDetector
import os
import icecream

logging.basicConfig(level=logging.DEBUG)
icecream.install()


DetectorNode.update_frequency = 1
model_info = ModelInformation(id='some_uuid', host='some_host', organization='zauberzeug',
                              project='test', version='1', categories=[Category(id='some_id', name='some_category_name')])
det = MockDetector()
det.init(model_info=model_info, model_root_path='')
node = DetectorNode(
    name='mocked detector',
    detector=det,
    uuid='85ef1a58-308d-4c80-8931-43d1f752f4f9',
)

# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    import uvicorn
    reload_dirs = ['./restart'] if os.environ.get('MANUAL_RESTART', None) \
        else ['./', './learning-loop-node', '/usr/local/lib/python3.7/site-packages/learning_loop_node']
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on',
                reload=True, use_colors=True, reload_dirs=reload_dirs)
