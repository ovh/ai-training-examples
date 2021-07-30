# Deploy a web service for YOLOv5 using Flask

The purpose of this tutorial is to show how to deploy a web service for YOLOv5 using your own weights generated after training a YOLOv5 model on your dataset.


## Instructions before using code

First, the tree structure of your folder should be as follows.

![image](tree_yolov5_web_service.png)

First, you have to create the folder named `models_train` and this is where you can store the weights generated after your trainings. You are free to put as many weight files as you want in the `models_train` folder.

**Then you can use the code above !**


## Test it locally (Optional)

Launch the following **docker command** to launch your application locally on your computer:

```console
docker run --rm -it -p 5000:5000 --user=42420:42420 yolov5_web:latest
```

> [!primary]
>
> The `-p 5000:5000` argument indicates that you want to execute a port rediction from the port **5000** of your local machine into the port **5000** of the docker container. The port **5000** is the default port used by **Flask** applications.
>


> :warning: Don't forget the `--user=42420:42420` argument if you want to simulate the exact same behavior that will occur on **AI TRAINING jobs**. It executes the docker container as the specific OVHcloud user (user **42420:42420**).
>

Once started, your application should be available on http://localhost:5000.


## Push the image into the shared registry

> :warning: The shared registry of AI Training should only be use for testing purpose. Please consider attaching your own docker registry. More information about this can be found [here][OVH Add private registry].
>

Find the adress of your shared registry by launching this command:

```console
ovhai registry list
```

Login on the shared registry with your usual openstack credencials:

```console
docker login -u <user> -p <password> <shared-registry-address>
```

Push the compiled image into the shared registry:

```console
docker tag yolov5_web:latest <shared-registry-address>/yolov5_web:latest
docker push <shared-registry-address>/yolov5_web:latest
```


## Launch the job

The following command starts a new job running your Flask application:

```console
ovhai job run --default-http-port 5000 --cpu 4 <shared-registry-address>/yolov5_web:latest
```

> [!primary]
>
> `--default-http-port 5000` indicates that the port to reach on the job url is the `5000`.
>

> [!primary]
>
> `--cpu 4` indicates that we request 4 cpu for that job.
>

> [!primary]
>
> Consider adding the `--unsecure-http` attribute if you want your application to be reachable without any authentication.
>
