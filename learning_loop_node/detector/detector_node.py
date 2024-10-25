import asyncio
import contextlib
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from threading import Thread
from typing import Dict, List, Optional, Union

import numpy as np
import socketio
from dacite import from_dict
from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from ..data_classes import Category, Context, Detections, DetectionStatus, ModelInformation, Shape
from ..data_classes.socket_response import SocketResponse
from ..data_exchanger import DataExchanger, DownloadError
from ..globals import GLOBALS
from ..helpers import environment_reader
from ..node import Node
from .detector_logic import DetectorLogic
from .inbox_filter.relevance_filter import RelevanceFilter
from .outbox import Outbox
from .rest import about as rest_about
from .rest import backdoor_controls
from .rest import detect as rest_detect
from .rest import model_version_control as rest_version_control
from .rest import operation_mode as rest_mode
from .rest import outbox_mode as rest_outbox_mode
from .rest import upload as rest_upload
from .rest.operation_mode import OperationMode


class DetectorNode(Node):

    def __init__(self, name: str, detector: DetectorLogic, uuid: Optional[str] = None, use_backdoor_controls: bool = False) -> None:
        super().__init__(name, uuid, 'detector', False)
        self.detector_logic = detector
        self.organization = environment_reader.organization()
        self.project = environment_reader.project()
        assert self.organization and self.project, 'Detector node needs an organization and an project'
        self.log.info('Using %s/%s', self.organization, self.project)
        self.operation_mode: OperationMode = OperationMode.Startup
        self.connected_clients: List[str] = []

        self.detection_lock = asyncio.Lock()

        self.outbox: Outbox = Outbox()
        self.data_exchanger = DataExchanger(
            Context(organization=self.organization, project=self.project),
            self.loop_communicator)

        self.relevance_filter: RelevanceFilter = RelevanceFilter(self.outbox)

        # NOTE: version_control controls the behavior of the detector node.
        # FollowLoop: the detector node will follow the loop and update the model if necessary
        # SpecificVersion: the detector node will update to a specific version, set via the /model_version endpoint
        # Pause: the detector node will not update the model
        self.version_control: rest_version_control.VersionMode = rest_version_control.VersionMode.Pause if os.environ.get(
            'VERSION_CONTROL_DEFAULT', 'follow_loop').lower() == 'pause' else rest_version_control.VersionMode.FollowLoop
        self.target_model: Optional[ModelInformation] = None
        self.loop_deployment_target: Optional[ModelInformation] = None

        self.include_router(rest_detect.router, tags=["detect"])
        self.include_router(rest_upload.router, prefix="")
        self.include_router(rest_mode.router, tags=["operation_mode"])
        self.include_router(rest_about.router, tags=["about"])
        self.include_router(rest_outbox_mode.router, tags=["outbox_mode"])
        self.include_router(rest_version_control.router, tags=["model_version"])

        if use_backdoor_controls or os.environ.get('USE_BACKDOOR_CONTROLS', '0').lower() in ('1', 'true'):
            self.include_router(backdoor_controls.router)

        self.setup_sio_server()

    async def soft_reload(self) -> None:
        # simulate init
        self.organization = environment_reader.organization()
        self.project = environment_reader.project()
        self.operation_mode = OperationMode.Startup
        self.connected_clients = []
        self.data_exchanger = DataExchanger(
            Context(organization=self.organization, project=self.project),
            self.loop_communicator)
        self.relevance_filter = RelevanceFilter(self.outbox)
        self.version_control = rest_version_control.VersionMode.Pause if os.environ.get(
            'VERSION_CONTROL_DEFAULT', 'follow_loop').lower() == 'pause' else rest_version_control.VersionMode.FollowLoop
        self.target_model = None
        # self.setup_sio_server()

        # simulate super().startup
        await self.loop_communicator.backend_ready()
        # await self.loop_communicator.ensure_login()
        self.set_skip_repeat_loop(False)
        self.socket_connection_broken = True
        await self.on_startup()

        # simulate startup
        await self.detector_logic.soft_reload()
        self.detector_logic.load_model()
        self.operation_mode = OperationMode.Idle

    async def on_startup(self) -> None:
        try:
            self.outbox.ensure_continuous_upload()
            self.detector_logic.load_model()
        except Exception:
            self.log.exception("error during 'startup'")
        self.operation_mode = OperationMode.Idle

    async def on_shutdown(self) -> None:
        try:
            await self.outbox.ensure_continuous_upload_stopped()
            for sid in self.connected_clients:
                # pylint: disable=no-member
                await self.sio.disconnect(sid)  # type:ignore
        except Exception:
            self.log.exception("error during 'shutdown'")

    async def on_repeat(self) -> None:
        try:
            await self._check_for_update()
        except Exception:
            self.log.exception("error during '_check_for_update'")

    def setup_sio_server(self) -> None:
        """The DetectorNode acts as a SocketIO server. This method sets up the server and defines the event handlers."""
        # pylint: disable=unused-argument

        # Initialize the Socket.IO server
        self.sio = socketio.AsyncServer(async_mode='asgi')
        # Initialize and mount the ASGI app
        self.sio_app = socketio.ASGIApp(self.sio, socketio_path='/socket.io')
        self.mount('/ws', self.sio_app)
        # Register event handlers

        self.log.info('>>>>>>>>>>>>>>>>>>>>>>> Setting up the SIO server')

        @self.sio.event
        async def detect(sid, data: Dict) -> Dict:
            self.log.debug('running detect via socketio')
            try:
                np_image = np.frombuffer(data['image'], np.uint8)
                det = await self.get_detections(
                    raw_image=np_image,
                    camera_id=data.get('camera-id', None) or data.get('mac', None),
                    tags=data.get('tags', []),
                    source=data.get('source', None),
                    autoupload=data.get('autoupload', None)
                )
                if det is None:
                    return {'error': 'no model loaded'}
                detection_dict = jsonable_encoder(asdict(det))
                self.log.debug('detect via socketio finished')
                return detection_dict
            except Exception as e:
                self.log.exception('could not detect via socketio')
                with open('/tmp/bad_img_from_socket_io.jpg', 'wb') as f:
                    f.write(data['image'])
                return {'error': str(e)}

        @self.sio.event
        async def info(sid) -> Union[str, Dict]:
            if self.detector_logic.is_initialized:
                return asdict(self.detector_logic.model_info)
            return 'No model loaded'

        @self.sio.event
        async def upload(sid, data: Dict) -> Optional[Dict]:
            '''upload an image with detections'''

            detection_data = data.get('detections', {})
            if detection_data and self.detector_logic.is_initialized:
                try:
                    detections = from_dict(data_class=Detections, data=detection_data)
                except Exception as e:
                    self.log.exception('could not parse detections')
                    return {'error': str(e)}
                detections = self.add_category_id_to_detections(self.detector_logic.model_info, detections)
            else:
                detections = Detections()

            tags = data.get('tags', [])
            tags.append('picked_by_system')

            source = data.get('source', None)

            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self.outbox.save, data['image'], detections, tags, source)
            except Exception as e:
                self.log.exception('could not upload via socketio')
                return {'error': str(e)}
            return None

        @self.sio.event
        def connect(sid, environ, auth) -> None:
            self.connected_clients.append(sid)

    async def _check_for_update(self) -> None:
        try:
            self.log.debug('Current operation mode is %s', self.operation_mode)
            try:
                await self.sync_status_with_learning_loop()
            except Exception:
                self.log.exception('Sync with learning loop failed (could not check for updates):')
                return

            if self.operation_mode != OperationMode.Idle:
                self.log.debug('not checking for updates; operation mode is %s', self.operation_mode)
                return

            self.status.reset_error('update_model')
            if self.target_model is None:
                self.log.debug('not checking for updates; no target model selected')
                return

            current_version = self.detector_logic._model_info.version if self.detector_logic._model_info is not None else None  # pylint: disable=protected-access

            if not self.detector_logic.is_initialized or self.target_model.version != current_version:
                self.log.info('Current model "%s" needs to be updated to %s',
                              current_version or "-", self.target_model.version)

                with step_into(GLOBALS.data_folder):
                    model_symlink = 'model'
                    target_model_folder = f'models/{self.target_model.version}'
                    shutil.rmtree(target_model_folder, ignore_errors=True)
                    os.makedirs(target_model_folder)

                    await self.data_exchanger.download_model(target_model_folder,
                                                             Context(organization=self.organization,
                                                                     project=self.project),
                                                             self.target_model.id, self.detector_logic.model_format)
                    try:
                        os.unlink(model_symlink)
                        os.remove(model_symlink)
                    except Exception:
                        pass
                    os.symlink(target_model_folder, model_symlink)
                    self.log.info('Updated symlink for model to %s', os.readlink(model_symlink))

                    self.detector_logic.load_model()
                    try:
                        await self.sync_status_with_learning_loop()
                    except Exception:
                        pass
                    # self.reload(reason='new model installed')

        except Exception as e:
            self.log.exception('check_for_update failed')
            msg = e.cause if isinstance(e, DownloadError) else str(e)
            self.status.set_error('update_model', f'Could not update model: {msg}')
            try:
                await self.sync_status_with_learning_loop()
            except Exception:
                pass

    async def sync_status_with_learning_loop(self) -> None:
        """Sync status of the detector with the Learning Loop.
        The Learning Loop will respond with the model info of the deployment target.
        If version_control is set to FollowLoop, the detector will update the target_model.
        Return if the communication was successful.

        Raises:
            Exception: If the communication with the Learning Loop failed.
        """

        if not self.sio_client.connected:
            self.log.info('Status sync failed: not connected')
            raise Exception('Status sync failed: not connected')

        try:
            current_model = self.detector_logic.model_info.version
        except Exception:
            current_model = None

        target_model_version = self.target_model.version if self.target_model else None

        status = DetectionStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            errors=self.status.errors,
            uptime=int((datetime.now() - self.startup_datetime).total_seconds()),
            operation_mode=self.operation_mode,
            current_model=current_model,
            target_model=target_model_version,
            model_format=self.detector_logic.model_format,
        )

        self.log.debug('sending status %s', status)
        response = await self.sio_client.call('update_detector', (self.organization, self.project, jsonable_encoder(asdict(status))))
        if not response:
            self.socket_connection_broken = True
            return

        socket_response = from_dict(data_class=SocketResponse, data=response)
        if not socket_response.success:
            self.socket_connection_broken = True
            self.log.error('Statusupdate failed: %s', response)
            raise Exception(f'Statusupdate failed: {response}')

        assert socket_response.payload is not None

        deployment_target_model_id = socket_response.payload['target_model_id']
        deployment_target_model_version = socket_response.payload['target_model_version']
        self.loop_deployment_target = ModelInformation(organization=self.organization, project=self.project,
                                                       host="", categories=[],
                                                       id=deployment_target_model_id,
                                                       version=deployment_target_model_version)

        if (self.version_control == rest_version_control.VersionMode.FollowLoop and
                self.target_model != self.loop_deployment_target):
            old_target_model_version = self.target_model.version if self.target_model else None
            self.target_model = self.loop_deployment_target
            self.log.info('After sending status. Target_model changed from %s to %s',
                          old_target_model_version, self.target_model.version)

    async def set_operation_mode(self, mode: OperationMode):
        self.operation_mode = mode
        try:
            await self.sync_status_with_learning_loop()
        except Exception as e:
            self.log.warning('Operation mode set to %s, but sync failed: %s', mode, e)

    def reload(self, reason: str):
        '''provide a cause for the reload'''

        self.log.info('########## reloading app because %s', reason)
        if os.path.isfile('/app/app_code/restart/restart.py'):
            subprocess.call(['touch', '/app/app_code/restart/restart.py'])
        elif os.path.isfile('/app/main.py'):
            subprocess.call(['touch', '/app/main.py'])
        elif os.path.isfile('/main.py'):
            subprocess.call(['touch', '/main.py'])
        else:
            self.log.error('could not reload app')

    async def get_detections(self,
                             raw_image: np.ndarray,
                             camera_id: Optional[str],
                             tags: List[str],
                             source: Optional[str] = None,
                             autoupload: Optional[str] = None) -> Detections:
        """ Main processing function for the detector node when an image is received via REST or SocketIO.
        This function infers the detections from the image, cares about uploading to the loop and returns the detections as a dictionary.
        Note: raw_image is a numpy array of type uint8, but not in the correct shape!
        It can be converted e.g. using cv2.imdecode(raw_image, cv2.IMREAD_COLOR)"""

        await self.detection_lock.acquire()
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(None, self.detector_logic.evaluate_with_all_info, raw_image, tags, source)
        self.detection_lock.release()

        fix_shape_detections(detections)
        n_bo, n_cl = len(detections.box_detections), len(detections.classification_detections)
        n_po, n_se = len(detections.point_detections), len(detections.segmentation_detections)
        self.log.debug('Detected: %d boxes, %d points, %d segs, %d classes', n_bo, n_po, n_se, n_cl)

        if autoupload is None or autoupload == 'filtered':  # NOTE default is filtered
            Thread(target=self.relevance_filter.may_upload_detections,
                   args=(detections, camera_id, raw_image, tags, source)).start()
        elif autoupload == 'all':
            Thread(target=self.outbox.save, args=(raw_image, detections, tags, source)).start()
        elif autoupload == 'disabled':
            pass
        else:
            self.log.error('unknown autoupload value %s', autoupload)
        return detections

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
        for classification_detection in detections.classification_detections:
            category_name = classification_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            classification_detection.category_id = category_id
        return detections

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


def fix_shape_detections(detections: Detections):
    # TODO This is a quick fix.. check how loop upload detections deals with this
    for seg_detection in detections.segmentation_detections:
        if isinstance(seg_detection.shape, Shape):
            points = ','.join([str(value) for p in seg_detection.shape.points for _,
                               value in asdict(p).items()])
            seg_detection.shape = points
