# Learning Loop Node

This Python library helps you to write your own Detection Nodes, Training Nodes and Converter Nodes for the Zauberzeug Learning Loop.

## General Usage

You can configure connection to our Learning Loop by specifying the following environment variables before starting:

- LOOP_HOST=learning-loop.ai
- LOOP_ORGANIZATION=<your organization>
- LOOP_PROJECT=<your project>
- LOOP_USERNAME=<your username>
- LOOP_PASSWORD=<your password>

## Detector Node

Detector Nodes are normally deployed on edge devices like robots or machinery but can also run in the cloud to provide backend services for an app or similar. These nodes register themself at the Learning Loop to make model deployments very easy. They also provide REST and Socket.io APIs to run inferences. By default the images will automatically used for active learning: high uncertain predictions will be submitted to the Learning Loop inbox.

## Trainer Node

Trainers fetch the images and anntoations from the Learning Loop to generate new and improved models.

- if the command line tool "jpeginfo" is installed, the downloader will drop corrupted images automatically

## Converter Node

A Conveter Node converts models from one format into another.

### How to test the operability?

Assumend there is a Converter Node which converts models of format 'format_a' into 'format_b'.
Upload a model with
`curl --request POST -F 'files=@my_model.zip' https://learning-loop.ai/api/zauberzeug/projects/demo/format_a`
The model should now be available for the format 'format_a'
`curl "https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_a"`

```
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
