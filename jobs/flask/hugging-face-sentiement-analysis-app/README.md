# Deploy a simple app for sentiment analysis with Hugging Face model using Flask

## Objective

The purpose of this tutorial is to show how to deploy a web service for sentiment analysis on text using your Hugging Face pretrained models.

In order to do this, you will use [Flask](https://flask.palletsprojects.com/en/2.0.x/), an open-source micro framework for web development in Python. You will also learn how to build and use a custom Docker image for a Flask application.

For more information on how to use Hugging Face models on your own dataset (for example: Tweets), refer to the following [documentation](PUT THE LINK).

## Requirements

* access to the [OVHcloud Control Panel](https://www.ovh.com/auth/?action=gotomanager)
* an AI Training project created inside a Public Cloud project
* a [user for AI Training][OVH Create user]
* [Docker][Docker Get Started] installed on your local computer
* some knowledge about building image and [Dockerfile][Dockerfile reference]

## Instructions

First, the tree structure of your folder should be as follows.

![image](images/NEW TREE STRUCTURE){.thumbnail}

Here we will mainly discuss how to write the `app.py` code, the `requirements.txt` file and the `Dockerfile`. If you want to see the whole code, please refer to the [GitHub](PUT THE LINK) repository.

### Write the Flask application

Create a Python file named `app.py`.

Inside that file, import your required modules:

```python
from flask import Flask, jsonify, render_template, request, make_response
import transformers
```

Create Flask app:

```python
app = Flask(__name__)
```

Load Hugging Face models:

Here a python dictionary is created to store the name (key) and the corresponding model (value).

You can choose the models you want from those offered by Hugging Face, depending on the language and type of text you want to analyse. Please refer to the [catalogue](https://huggingface.co/models) of models.

```python
# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {"RoBERTa" : transformers.pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english"), "BERT" : transformers.pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")}

# create a list of keys to use them in the select part of the html code
listOfKeys = []
for key in dictOfModels :
        listOfKeys.append(key)     # put all the keys in the listOfKeys
```

Write the inference function:

```python
def get_prediction(message,model):
    # inference
    results = model(message)  
    return results
```

Define the GET method:

```python
@app.route('/', methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template("home.html", len = len(listOfKeys), listOfKeys = listOfKeys)
```

Define the POST method:

```python
@app.route('/', methods=['POST'])
def predict():
    message = request.form['message']
    # choice of the model
    results = get_prediction(message, dictOfModels[request.form.get("model_choice")])
    print(f'User selected model : {request.form.get("model_choice")}')
    my_prediction = f'The feeling of this text is {results[0]["label"]} with probability of {results[0]["score"]*100}%.'
    return render_template('result.html', text = f'{message}', prediction = my_prediction)
```

Start your app:

```python
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
```

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

> [!primary]
>
> The dot `.` argument indicates that your build context (place of the **Dockerfile** and other needed files) is the current directory.
>

> [!primary]
>
> The `-t` argument allows you to choose the identifier to give to your image. Usually image identifiers are composed of a **name** and a **version tag** `<name>:<version>`. For this example we chose **sentiment_analysis_app:latest**.
>

### Test it locally (optional)

Launch the following **Docker command** to launch your application locally on your computer:

```console
docker run --rm -it -p 5000:5000 --user=42420:42420 sentiment_analysis_app:latest
```

> [!primary]
>
> The `-p 5000:5000` argument indicates that you want to execute a port redirection from the port **5000** of your local machine into the port **5000** of the Docker container. The port **5000** is the default port used by **Flask** applications.
>

> [!warning]
>
> Don't forget the `--user=42420:42420` argument if you want to simulate the exact same behaviour that will occur on **AI TRAINING jobs**. It executes the Docker container as the specific OVHcloud user (user **42420:42420**).
>

Once started, your application should be available on `http://localhost:5000`.

### Push the image into the shared registry

> [!warning]
>
> The shared registry of AI Training should only be used for testing purpose. Please consider attaching your own Docker registry. More information about this can be found [here][OVH Add private registry].
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

> [!primary]
>
> `--default-http-port 5000` indicates that the port to reach on the job URL is the `5000`.
>

> [!primary]
>
> `--cpu 4` indicates that we request 4 CPUs for that job.
>

> [!primary]
>
> Consider adding the `--unsecure-http` attribute if you want your application to be reachable without any authentication.
>
