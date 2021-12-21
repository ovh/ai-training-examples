# Deploy an app for sentiment analysis with Hugging Face models using Flask

The purpose of this tutorial is to show how to deploy a web service for sentiment analysis on text using your Hugging Face pretrained models.

## Instructions before using code

First, the tree structure of your folder should be as follows.

![image](tree-flask-app.png)

Find more information about the Flask application [here](https://flask.palletsprojects.com/en/2.0.x/quickstart/#a-minimal-application) to get ready to use it.

### Write the requirements.txt file for the application

The `requirements.txt` file will allow us to write all the modules needed to make our application work. This file will be useful when writing the `Dockerfile`.

```console
Flask==1.1.2

transformers
```

### Write the Dockerfile for the application

Your Dockerfile should start with the the `FROM` instruction indicating the parent image to use. In our case we choose to start from a pytorch image:

```console
FROM pytorch/pytorch
```

Create the home directory and add your files to it:

```console
WORKDIR /workspace
ADD . /workspace
```

Install the `requirements.txt` file which contains your needed Python modules using a `pip install ...` command:

```console
RUN pip install -r requirements.txt
```

Define your default launching command to start the application:

```console
CMD [ "python" , "/workspace/app.py" ]
```

Give correct access rights to **ovhcloud user** (`42420:42420`):

```console
RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
```

### Build the Docker image from the Dockerfile

Launch the following command from the **Dockerfile** directory to build your application image:

```console
docker build . -t sentiment_analysis_app:latest
```

> :heavy_exclamation_mark: The dot `.` argument indicates that your build context (place of the **Dockerfile** and other needed files) is the current directory.
>

> :heavy_exclamation_mark: The `-t` argument allows you to choose the identifier to give to your image. Usually image identifiers are composed of a **name** and a **version tag** `<name>:<version>`. For this example we chose **sentiment_analysis_app:latest**.
>

### Test it locally (optional)

Launch the following **Docker command** to launch your application locally on your computer:

```console
docker run --rm -it -p 5000:5000 --user=42420:42420 sentiment_analysis_app:latest
```

> :heavy_exclamation_mark: The `-p 5000:5000` argument indicates that you want to execute a port redirection from the port **5000** of your local machine into the port **5000** of the Docker container. The port **5000** is the default port used by **Flask** applications.
>

> :warning: Don't forget the `--user=42420:42420` argument if you want to simulate the exact same behaviour that will occur on **AI TRAINING jobs**. It executes the Docker container as the specific OVHcloud user (user **42420:42420**).
>

Once started, your application should be available on `http://localhost:5000`.

### Push the image into the shared registry

> :warning: The shared registry of AI Training should only be used for testing purpose. Please consider attaching your own Docker registry. More information about this can be found [here][OVH Add private registry].
>

Find the adress of your shared registry by launching this command:

```console
ovhai registry list
```

Login on the shared registry with your usual openstack credentials:

```console
docker login -u <user> -p <password> <shared-registry-address>
```

Push the compiled image into the shared registry:

```console
docker tag sentiment_analysis_app:latest <shared-registry-address>/sentiment_analysis_app:latest
docker push <shared-registry-address>/sentiment_analysis_app:latest
```

### Launch the job

The following command starts a new job running your Flask application:

```console
ovhai job run --default-http-port 5000 --cpu 4 <shared-registry-address>/sentiment_analysis_app:latest
```

> :heavy_exclamation_mark: `--default-http-port 5000` indicates that the port to reach on the job URL is the `5000`.
>

> :heavy_exclamation_mark: `--cpu 4` indicates that we request 4 CPUs for that job.
>

> :heavy_exclamation_mark: Consider adding the `--unsecure-http` attribute if you want your application to be reachable without any authentication.
>
