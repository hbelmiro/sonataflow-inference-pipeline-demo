# Deploying YOLOv5 on ODH

## ODH Setup + Install

1. Create two projects, one called `opendatahub` and the other `model-project`. (These can have any names, but let's use these for reference):

    ```bash
    oc new-project opendatahub
    oc new-project model-project
    ```

2. Install ODH Operator into `All namespaces` via OperatorHub

3. Install the default ODH KFDef into the `opendatahub` project, via Installed Operators -> OpenDataHub -> KFDef -> "Create KFDef"

## Model Namespace Setup

1. Switch to your model-project

    ```bash
    oc project model-project
    ```

2. Label the project for modelmesh compatibility

    ```bash
    oc label namespace model-project "modelmesh-enabled=true" --overwrite=true
    ```

## Model Deployment

1. Deploy the minio container

    ```bash
    oc apply -f trustyai_yolo_minio.yaml
    ```

2. Deploy the OVMS runtime

    ```bash
    oc apply -f ovms-1.x.yaml
    ```

3. Deploy the YOLO model

    ```bash
    oc apply -f model_yolo.yaml
    ```

## Accessing the Model

Find the model inference route in the Routes page within the `model-project` project. It should look something like: `https://yolo-model-model-project.apps.xyz…/v2/models/yolo-model`
