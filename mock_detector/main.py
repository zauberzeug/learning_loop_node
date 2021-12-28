import logging
logging.basicConfig(level=logging.DEBUG)
import backdoor_controls
from mock_detector import MockDetector
import uvicorn
from learning_loop_node.detector.detector_node import DetectorNode
import os
import icecream
icecream.install()


DetectorNode.update_frequency = 1
node = DetectorNode(
    name='mocked detector',
    detector=MockDetector(),
    uuid='85ef1a58-308d-4c80-8931-43d1f752f4f9',
)

# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    reload_dirs = ['./restart'] if os.environ.get('MANUAL_RESTART', None) \
        else ['./', './learning-loop-node', '/usr/local/lib/python3.7/site-packages/learning_loop_node']
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True, reload_dirs=reload_dirs)
