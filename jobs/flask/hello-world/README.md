# Deploy a simple app with Flask

The purpose of this tutorial is to show how to deploy a simple app with **Flask**.

## Instructions before using code

First, the tree structure of your folder should be as follows.

![image](tree-flask-app.png)

### Build the Docker image from the Dockerfile

Launch the following command from the **Dockerfile** directory to build your application image:

```console
docker build . -t flask-app:latest
```

> [!primary]
>
> The dot `.` argument indicates that your build context (place of the **Dockerfile** and other needed files) is the current directory.
>

> [!primary]
>
> The `-t` argument allows you to choose the identifier to give to your image. Usually image identifiers are composed of a **name** and a **version tag** `<name>:<version>`. For this example we chose **flask-app:latest**.
>

### Test it locally (optional)

Launch the following **docker command** to launch your application locally on your computer:

```console
docker run --rm -it -p 5000:5000 --user=42420:42420 flask-app:latest
```

> :heavy_exclamation_mark: The `-p 5000:5000` argument indicates that you want to execute a port rediction from the port **5000** of your local machine into the port **5000** of the docker container. The port **5000** is the default port used by **Flask** applications.
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
docker tag flask-app:latest <shared-registry-address>/flask-app:latest
docker push <shared-registry-address>/flask-app:latest
```

## Launch the job

The following command starts a new job running your Flask application:

```console
ovhai job run --default-http-port 5000 --cpu 1 <shared-registry-address>/flask-app:latest
```

> :heavy_exclamation_mark: `--default-http-port 5000` indicates that the port to reach on the job url is the `5000`.
>

> :heavy_exclamation_mark: `--cpu 1` indicates that we request 4 cpu for that job.
>

> :heavy_exclamation_mark: Consider adding the `--unsecure-http` attribute if you want your application to be reachable without any authentication.
>
