import socketio
import requests


class Events(socketio.AsyncClientNamespace):

    def __init__(self, hostname: str):
        super().__init__('/')
        self.hostname = hostname

    def on_save(self, model):
        print('---- saving model', model['id'], flush=True)
        if not hasattr(self, 'get_weightfile'):
            return 'node does not provide a get_weightfile function'

        ogranization = model['context']['organization']
        project = model['context']['project']
        uri_base = f'http://{self.hostname}/api/{ogranization}/projects/{project}'
        response = requests.put(
            f'{uri_base}/models/{model["id"]}/file',
            files={'data': self.get_weightfile(model)}
        )
        if response.status_code == 200:
            return True
        else:
            return response.json()['detail']
