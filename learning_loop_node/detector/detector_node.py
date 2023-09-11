import asyncio
import contextlib
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from threading import Thread
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from dacite import from_dict
from fastapi.encoders import jsonable_encoder
from fastapi_socketio import SocketManager
from socketio import AsyncClient

from .. import environment_reader
from ..data_classes import (Category, Context, Detections, DetectionStatus,
                            ModelInformation, NodeState, Shape)
from ..data_exchanger import DataExchanger, DownloadError
from ..globals import GLOBALS
from ..inbox_filter import RelevanceFilter
from ..node import Node
from ..socket_response import SocketResponse
from .detector_logic import DetectorLogic
from .outbox import Outbox
from .rest import detect as rest_detect
from .rest import operation_mode as rest_mode
from .rest import upload as rest_upload
from .rest.operation_mode import OperationMode


class DetectorNode(Node):
    update_frequency = 10

    def __init__(self, name: str, detector: DetectorLogic, uuid: Optional[str] = None):
        super().__init__(name, uuid)
        self.detector_logic = detector
        self.organization = environment_reader.organization()
        self.project = environment_reader.project()
        assert self.organization and self.project, 'Detector node needs an organization and an project'
        self.log.info(f'Using {self.organization}/{self.project}')
        self.operation_mode: OperationMode = OperationMode.Startup
        self.connected_clients: List[str] = []

        self.outbox: Outbox = Outbox()
        self.data_exchanger = DataExchanger(
            Context(organization=self.organization, project=self.project),
            self.loop_communicator)

        self.relevance_filter: RelevanceFilter = RelevanceFilter(self.outbox)
        self.target_model = None

        self.include_router(rest_detect.router, tags=["detect"])
        self.include_router(rest_upload.router, prefix="")
        self.include_router(rest_mode.router, tags=["operation_mode"])

        self.setup_sio_server()

    async def on_startup(self):
        try:
            self.outbox.start_continuous_upload()
            self.detector_logic.load_model()
        except Exception:
            self.log.exception("error during 'startup'")
        self.operation_mode = OperationMode.Idle

    async def on_shutdown(self):
        try:
            self.outbox.stop_continuous_upload()
            for sid in self.connected_clients:
                await self.sio.disconnect(sid)  # type:ignore pylint: disable=no-member
        except Exception:
            self.log.exception("error during 'shutdown'")

    async def on_repeat(self):
        try:
            await self._check_for_update()
        except Exception:
            self.log.exception("error during '_check_for_update'")

    def setup_sio_server(self):
        """The DetectorNode acts as a SocketIO server. This method sets up the server and defines the event handlers."""

        # pylint: disable=unused-argument

        async def _detect(sid, data) -> Dict:
            self.log.info('running detect via socketio')
            try:
                np_image = np.frombuffer(data['image'], np.uint8)
                det = await self.get_detections(
                    raw_image=np_image,
                    camera_id=data.get('camera-id', None) or data.get('mac', None),
                    tags=data.get('tags', []),
                    autoupload=data.get('autoupload', None),
                )
                if det is None:
                    return {'error': 'no model loaded'}
                self.log.info('detect via socketio finished')
                return det
            except Exception as e:
                self.log.exception('could not detect via socketio')
                with open('/tmp/bad_img_from_socket_io.jpg', 'wb') as f:
                    f.write(data['image'])
                return {'error': str(e)}

        async def _info(sid) -> Union[str, Dict]:
            if self.detector_logic.is_initialized:
                return asdict(self.detector_logic.model_info)
            return 'No model loaded'

        async def _upload(sid, data) -> Optional[Dict]:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self.outbox.save, data['image'], Detections(), ['picked_by_system'])
            except Exception as e:
                self.log.exception('could not upload via socketio')
                return {'error': str(e)}

        def _connect(sid, environ, auth):
            self.connected_clients.append(sid)

        self.sio_server = SocketManager(app=self)
        self.sio_server.on('detect', _detect)
        self.sio_server.on('info', _info)
        self.sio_server.on('upload', _upload)
        self.sio_server.on('connect', _connect)

    async def _check_for_update(self):
        if self.operation_mode == OperationMode.Startup:
            return
        try:
            self.log.info(f'Current operation mode is {self.operation_mode}')
            update_to_model_id = await self.send_status()
            if not update_to_model_id:
                self.log.info('could not check for updates')
                return
            if self.detector_logic.is_initialized:
                self.log.info(
                    f'Current model: {self.detector_logic.model_info.version} with id {self.detector_logic.model_info.id}')
            else:
                self.log.info('no model loaded')
            if self.operation_mode != OperationMode.Idle:
                self.log.info(f'not checking for updates; operation mode is {self.operation_mode}')
                return

            self.status.reset_error('update_model')
            if self.target_model is None:
                self.log.info('not checking for updates; no target model selected')
                return

            self.log.info('going to check for new updates')
            if not self.detector_logic.is_initialized or self.target_model != self.detector_logic.model_info.version:
                cur_model = self.detector_logic.model_info.version if self.detector_logic.is_initialized else "-"
                self.log.info(f'Current model "{cur_model}" needs to be updated to {self.target_model}')
                with step_into(GLOBALS.data_folder):
                    model_symlink = 'model'
                    target_model_folder = f'models/{self.target_model}'
                    shutil.rmtree(target_model_folder, ignore_errors=True)
                    os.makedirs(target_model_folder)

                    await self.data_exchanger.download_model(
                        target_model_folder,
                        Context(organization=self.organization, project=self.project),
                        update_to_model_id,
                        self.detector_logic.model_format
                    )
                    try:
                        os.unlink(model_symlink)
                        os.remove(model_symlink)
                    except Exception:
                        pass
                    os.symlink(target_model_folder, model_symlink)
                    self.log.info(f'Updated symlink for model to {os.readlink(model_symlink)}')

                    self.detector_logic.load_model()
                    await self.send_status()
                    # self.reload(reason='new model installed')
            else:
                self.log.info('Versions are identic. Nothing to do.')
        except Exception as e:
            self.log.exception('check_for_update failed')
            msg = e.cause if isinstance(e, DownloadError) else str(e)
            self.status.set_error('update_model', f'Could not update model: {msg}')
            await self.send_status()

    async def send_status(self) -> Union[str, Literal[False]]:
        if not self.sio_client.connected:
            self.log.info('could not send status -- we are not connected to the Learning Loop')
            return False

        status = DetectionStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            errors=self.status.errors,
            uptime=int((datetime.now() - self.startup_time).total_seconds()),
            operation_mode=self.operation_mode,
            current_model=self.detector_logic.model_info.version if self.detector_logic.is_initialized else None,
            target_model=self.target_model,
            model_format=self.detector_logic.model_format,
        )

        self.log.info(f'sending status {status}')
        response = await self.sio_client.call('update_detector', (self.organization, self.project, jsonable_encoder(asdict(status))))
        assert response is not None
        socket_response = from_dict(data_class=SocketResponse, data=response)
        if not socket_response.success:
            self.log.error(f'Statusupdate failed: {response}')
            return False

        assert socket_response.payload is not None
        self.target_model = socket_response.payload['target_model_version']
        self.log.info(f'After sending status. Target_model is {self.target_model}')
        return socket_response.payload['target_model_id']

    async def get_state(self):
        return NodeState.Online  # NOTE At the moment only trainer-nodes use a meaningful state

    async def set_operation_mode(self, mode: OperationMode):
        self.operation_mode = mode
        await self.send_status()

    def reload(self, reason: str):
        '''provide a cause for the reload'''

        self.log.info(f'########## reloading app because {reason}')
        if os.path.isfile('/app/restart/restart.py'):
            subprocess.call(['touch', '/app/restart/restart.py'])
        elif os.path.isfile('/app/main.py'):
            subprocess.call(['touch', '/app/main.py'])
        elif os.path.isfile('/main.py'):
            subprocess.call(['touch', '/main.py'])
        else:
            self.log.error('could not reload app')

    async def get_detections(self, raw_image, camera_id: Optional[str], tags: List[str], autoupload: Optional[str] = None) -> Optional[Dict]:

        loop = asyncio.get_event_loop()
        detections: Detections = await loop.run_in_executor(None, self.detector_logic.evaluate, raw_image)

        for seg_detection in detections.segmentation_detections:
            if isinstance(seg_detection.shape, Shape):
                shapes = ','.join([str(value) for p in seg_detection.shape.points for _,
                                   value in asdict(p).items()])
                seg_detection.shape = shapes  # TODO This seems to be a quick fix.. check how loop upload detections deals with this

        n_bo, n_cl = len(detections.box_detections), len(detections.classification_detections)
        n_po, n_se = len(detections.point_detections), len(detections.segmentation_detections)
        self.log.info(f'detected:{n_bo} boxes, {n_po} points, {n_se} segs, {n_cl} classes')

        if camera_id is not None:
            tags.append(camera_id)
        if autoupload is None or autoupload == 'filtered':  # NOTE default is filtered
            Thread(target=self.relevance_filter.may_upload_detections,
                   args=(detections, camera_id, raw_image, tags)).start()
        elif autoupload == 'all':
            Thread(target=self.outbox.save, args=(raw_image, detections, tags)).start()
        elif autoupload == 'disabled':
            pass
        else:
            self.log.error(f'unknown autoupload value {autoupload}')
        return jsonable_encoder(asdict(detections))

    async def upload_images(self, images: List[bytes]):
        loop = asyncio.get_event_loop()
        for image in images:
            await loop.run_in_executor(None, self.outbox.save, image, Detections(), ['picked_by_system'])

    def add_category_id_to_detections(self, model_info: ModelInformation, detections: Detections):
        def find_category_id_by_name(categories: List[Category], category_name: str):
            category_id = [category.id for category in categories if category.name == category_name]
            return category_id[0] if category_id else ''

        for box_detection in detections.box_detections:
            category_name = box_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            box_detection.category_id = category_id
        for point_detection in detections.point_detections:
            category_name = point_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            point_detection.category_id = category_id
        for segmentation_detection in detections.segmentation_detections:
            category_name = segmentation_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            segmentation_detection.category_id = category_id
        return detections

    def get_node_type(self):
        return 'detector'

    def register_sio_events(self, sio_client: AsyncClient):
        pass


@contextlib.contextmanager
def step_into(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)
