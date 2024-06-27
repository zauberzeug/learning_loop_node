# Learning Loop Node

This Python library helps to write Nodes that interact with the Zauberzeug Learning Loop. There are 4 types of Nodes:

| Type      | Purpose                                              |
| --------- | ---------------------------------------------------- |
| Trainer   | Runs training using the latest training data         |
| Detector  | Loads latest models from Loop and performs inference |
| Annotator | Used for custom annotation inside the Loop           |
| Converter | Converts between different model formats             |

## General Usage

To start a node you have to implement the logic by inheriting from the corresponding base logic class. We provide samples in the 'mock' folders and recommend to follow that scheme. A complete trainer and detector example can be found [here](https://github.com/zauberzeug/yolov5_node).

#### Environment variables

You can configure connection to our Learning Loop by specifying the following environment variables before starting:

| Name                    | Alias        | Purpose                                                      | Required by          |
| ----------------------- | ------------ | ------------------------------------------------------------ | -------------------- |
| LOOP_HOST               | HOST         | Learning Loop address (e.g. learning-loop.ai)                | all                  |
| LOOP_USERNAME           | USERNAME     | Learning Loop user name                                      | all besides Detector |
| LOOP_PASSWORD           | PASSWORD     | Learning Loop password                                       | all besides Detector |
| LOOP_SSL_CERT_PATH      | -            | Path to the SSL certificate                                  | all (opt.)           |
| LOOP_ORGANIZATION       | ORGANIZATION | Organization name                                            | Detector             |
| LOOP_PROJECT            | PROJECT      | Project name                                                 | Detector             |
| MIN_UNCERTAIN_THRESHOLD | PROJECT      | smallest confidence (float) at which auto-upload will happen | Detector             |
| MAX_UNCERTAIN_THRESHOLD | PROJECT      | largest confidence (float) at which auto-upload will happen  | Detector             |
| INFERENCE_BATCH_SIZE    | -            | Batch size of trainer when calculating detections            | Trainer (opt.)       |
| RESTART_AFTER_TRAINING  | -            | Restart the trainer after training (set to 1)                | Trainer (opt.)       |
| KEEP_OLD_TRAININGS      | -            | Do not delete old trainings (set to 1)                       | Trainer (opt.)       |

#### Testing

We use github actions for CI. Tests can also be executed locally by running
`LOOP_HOST=XXXXXXXX LOOP_USERNAME=XXXXXXXX LOOP_PASSWORD=XXXXXXXX python -m pytest -v`  
from learning_loop_node/learning_loop_node

## Detector Node

Detector Nodes are normally deployed on edge devices like robots or machinery but can also run in the cloud to provide backend services for an app or similar. These nodes register themself at the Learning Loop. They provide REST and Socket.io APIs to run inference on images. The processed images can automatically be used for active learning: e.g. uncertain predictions will be send to the Learning Loop.

### Running Inference

Images can be send to the detector node via socketio or rest.
The later approach can be used via curl,

Example usage:

`curl --request POST -F 'file=@test.jpg' localhost:8004/detect`

Where 8804 is the specified port in this example.
You can additionally provide the following camera parameters:

- `autoupload`: configures auto-submission to the learning loop; `filtered` (default), `all`, `disabled` (example curl parameter `-H 'autoupload: all'`)
- `camera-id`: a string which groups images for submission together (example curl parameter `-H 'camera-id: front_cam'`)

The detector also has a sio **upload endpoint** that can be used to upload images and detections to the learning loop. The function receives a json dictionary, with the following entries:

- `image`: the image data in jpg format
- `tags`: a list of strings. If not provided the tag is `picked_by_system`
- `detections`: a dictionary representing the detections. UUIDs for the classes are automatically determined based on the category names. This field is optional. If not provided, no detections are uploaded.

The endpoint returns None if the upload was successful and an error message otherwise.

### Changing the outbox mode

If the autoupload is set to `all` or `filtered` (selected) images and the corresponding detections are saved on HDD (the outbox). A background thread will upload the images and detections to the Learning Loop. The outbox is located in the `outbox` folder in the root directory of the node. The outbox can be cleared by deleting the files in the folder.

The continuous upload can be stopped/started via a REST enpoint:

Example Usage:

- Enable upload: `curl -X PUT -d "continuous_upload" http://localhost/outbox_mode`
- Disable upload: `curl -X PUT -d "stopped" http://localhost/outbox_mode`

The current state can be queried via a GET request:
`curl http://localhost/outbox_mode`

### Explicit upload

The detector has a REST endpoint to upload images (and detections) to the Learning Loop. The endpoint takes a POST request with the image and optionally the detections. The image is expected to be in jpg format. The detections are expected to be a json dictionary. Example:

`curl -X POST -F 'files=@test.jpg' "http://localhost:/upload"`

## Trainer Node

Trainers fetch the images and anntoations from the Learning Loop to train new models.

- if the command line tool "jpeginfo" is installed, the downloader will drop corrupted images automatically

## Converter Node

A Conveter Node converts models from one format into another.

## Annotator Node

...

#### Test operability

Assumend there is a Converter Node which converts models of format 'format_a' into 'format_b'.
Upload a model with
`curl --request POST -F 'files=@my_model.zip' https://learning-loop.ai/api/zauberzeug/projects/demo/format_a`
The model should now be available for the format 'format_a'
`curl "https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_a"`

````

{
"models": [
{
"id": "3c20d807-f71c-40dc-a996-8a8968aa5431",
"version": "4.0",
"formats": [
"format_a"
],
"created": "2021-06-01T06:28:21.289092",
"comment": "uploaded at 2021-06-01 06:28:21.288442",
...
}
]
}

```

but not in the format_b
`curl "https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_b"`

```

{
"models": []
}

```

Connect the Node to the Learning Loop by simply starting the container.
After a short time the converted model should be available as well.
`curl https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_b`

```

{
"models": [
{
"id": "3c20d807-f71c-40dc-a996-8a8968aa5431",
"version": "4.0",
"formats": [
"format_a",
"format_b",
],
"created": "2021-06-01T06:28:21.289092",
"comment": "uploaded at 2021-06-01 06:28:21.288442",
...
}
]
}

```

## About Models (the currency between Nodes)

- Models are packed in zips and saved on the Learning Loop (one for each format)
- Nodes and users can upload and download models with which they want to work
- In each zip there is a file called `model.json` which contains the metadata to interpret the other files in the package
- for base models (pretrained models from external sources) no `model.json` has to be sent, ie. these models should simply be zipped in such a way that the respective trainer can work with them.
- the loop adds or corrects the following properties in the `model.json` after receiving; it also creates the file if it is missing:
  - `host`: uri to the loop
  - `organization`: the ID of the organization
  - `project`: the id of the project
  - `version`: the version number that the loop assigned for this model (e.g. 1.3)
  - `id`: the model UUID (currently not needed by anyone, since host, org, project, version clearly identify the model)
  - `format`: the format e.g. yolo, tkdnn, yolor etc.
- Nodes add properties to `model.json`, which contains all the information which are needed by subsequent nodes. These are typically the properties:
  - `resolution`: resolution in which the model expects images (as `int`, since the resolution is mostly square - later, ` resolution_x`` resolution_y ` would also be conceivable or `resolutions` to give a list of possible resolutions)
  - `categories`: list of categories with name, id, (later also type), in the order in which they are used by the model -- this is neccessary to be robust about renamings
```
````
