import os

from app_code.mock_detector import MockDetectorFactory

from learning_loop_node import DetectorNode

node = DetectorNode(
    name='mocked detector',
    detector_factory=MockDetectorFactory(),
    uuid='85ef1a58-308d-4c80-8931-43d1f752f4f9',
    use_backdoor_controls=True,
)

if __name__ == "__main__":
    import uvicorn
    reload_dirs = ['./app_code/restart'] if os.environ.get('MANUAL_RESTART', None) \
        else ['./app_code', './learning-loop-node', '/usr/local/lib/python3.11/site-packages/learning_loop_node']
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on',
                reload=True, use_colors=True, reload_dirs=reload_dirs)
