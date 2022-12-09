# README

# Documentation chat bot to the cloud

I decided here to create a chat bot from scratch with the framework rasa. My first chatbot will an assistant chatbot for a customer. It is a kind of help where a client want for example information with a command he made on the website or want to complaint. So I have to found some data about this topic.

The problem was the data was in a format where we can’t directly use it. Rasa framework use “.yml” files. Most of the dataset I found are in “.csv” files. To transform the files, I use some files organize in the folder “process_data”.

Here is the main file of data in rasa. It is called by default the “nlu.yml” files. In this file, we found specific intent with the name. For example, an intent can be “ask_for_a_command”. In one intent, we have examples of how the customer will be able to speak about the intent. An example of “ask_for_a_command” could be “Hi, i want to know if my command has been delivered.” More we provide some examples of intents, more the rasa chat bot will be efficient and useful.

After provided all of the examples for the all of the intents, we have to provide a response to the intent of the customer. To do this, we also use a file called process_domain.py. This file will created the file “domain.yml” file. We can found the response of the intents provided. In my example, the response of the chatbot will just be “Hi we detect the intent : ask_for_a_command” if the user ask for a command.

The last file to create is the “rules.yml”. This one specify what the chat bot do every time he found a specific intent. It can be for example when a user say hello respond with hello. In this file, I decided to say to the chatbot, if you found a intent respond with I detect this intent. The file to do this is named “process_rules.py”.

Here is a small diagram to understand what contains the nlu file.

[![](https://mermaid.ink/img/pako:eNpVkdtqwkAQhl9l2KsW1La3QRSPILS9sSdIRCbupFncQ9hsqqn67h2NtXZgYZmZ_5vTTqycJBGJT49FDi_jxALbIH5-fO3URkOmNC2g3e7tR84GVLbcw_BmZgPZABEMyvUyc36Jy5UzBq28bQDDk2SqfBmAtmgKTXsYxzPYIOuCg7V1G-jepb6nMjA1nOWNK8cSUiILkrT6Ik9ycY3tsMHklzqJR2ihdhUE0hoMNQxm2sqk5MFlF3r_H-cj5H_NTeP3nCsB8mNtmTsqod9lEszAoCQIOV1AGnmwDdG6sxAtYcgbVJLXuDvyE8GphhIR8VdShpUOiWhdhd7QK0w1lcecXdNTIjLe8Fx9n4UP98U2EU3skNgD18EquHltVyIKvqKWqAqJgcYK-XpGRBnqkr0kVXD-qbnr6byHHwI1oAA)](https://mermaid.live/edit#pako:eNpVkdtqwkAQhl9l2KsW1La3QRSPILS9sSdIRCbupFncQ9hsqqn67h2NtXZgYZmZ_5vTTqycJBGJT49FDi_jxALbIH5-fO3URkOmNC2g3e7tR84GVLbcw_BmZgPZABEMyvUyc36Jy5UzBq28bQDDk2SqfBmAtmgKTXsYxzPYIOuCg7V1G-jepb6nMjA1nOWNK8cSUiILkrT6Ik9ycY3tsMHklzqJR2ihdhUE0hoMNQxm2sqk5MFlF3r_H-cj5H_NTeP3nCsB8mNtmTsqod9lEszAoCQIOV1AGnmwDdG6sxAtYcgbVJLXuDvyE8GphhIR8VdShpUOiWhdhd7QK0w1lcecXdNTIjLe8Fx9n4UP98U2EU3skNgD18EquHltVyIKvqKWqAqJgcYK-XpGRBnqkr0kVXD-qbnr6byHHwI1oAA)

# Create the docker image

Here is the commande to create the Docker Image and to push it on my private docker directory. The image will be by default public on your private directory.

```bash
docker build . -t <yourdockerhubId>/big_chatbot:latest
docker push <yourdockerhubId>/big_chatbot:latest
```

# Create 2 tokens

One token is for operate the notebook and all of the work related to the rasa chatbot, the other to see the work I have done. One token is in read only and the other is only for the administrator.

```bash
ovhai token create -l model=rasabotRO --role read token-RO-chatbot
ovhai token create -l model=rasabotRW --role operator token-RW-chatbot
```

For each line, a value of the token is written. Don’t forget to save it because you can’t get it after this. Now when we create all of are product in OVH, we just have to add the label as options to have only access with the tokens and not with users. With this option, our connection is more secure.

# Create a VS code notebook and connect to remote on it (Not neccessarily).

Here is the command to create the notebook. We add two tokens. One for RO only and the other for read and write.

``` bash
ovhai notebook run conda vscode \
	--name vscode-ovh-chatbot \
	--framework-version conda-py39-cuda11.2-v22-4 \
	--volume <data-to-train-container>@GRA/data:/workspace/data:RO:cache \
	--volume <model-output-container>@GRA/:/workspace/trained-models:RW \
	--volume https://github.com/Victor2103/rasa_chatbot.git:/workspace/public-repo-git:RO \
	--cpu 10 \
	--token <token> \
	--label model=rasabotRO \
	-s ~/.ssh/id_rsa.pub
```

You can also of course stop the notebook when you want. It is really advice to stop the notebook when you don’t using it. With the CLI command, you can restart the notebook when you want. To do this, get the ID of your notebook with “ovhai notebook ls” and then run

```bash
ovhai notebook stop <jobid>
```

To re run the notebook just launch

```bash
ovhai notebook start --token <token> <jobid>
```

Once your notebook is running, open a terminal and go into the folder public-repo-git. Then install pip with conda and install the requirements for rasa with the file requirements_rasa.txt. Here are the command to do so. You can after this train the model. To do this, you can connect in a terminal by ssh in your machine or connect on the browser with the token we create before.

```bash
ssh <notebook-id>@<region>.training.ai.cloud.ovh.net
```

Then just run this in the terminal to train the model. 

```bash
conda install pip
python3 -m pip install --no-cache-dir -r requirements_rasa.txt
cd rasa_bot/
rasa train
```

If you want to save the model in your object storage, put your model on the folder saved_model. Then, he will be available on the private container <writenamecontainer> at the root. 
Once you've run your model, you can speak with him. To do so, just run in a terminal this command : 

```bash
rasa shell
```

Of course, with this command, your chatbot will not have the functionnality provided. To have all of the functionnalities and a fonctionnal chatbot, follow the end of the tutorial ! 

# Play with a Jupyter notebook (optional). 

You're more familiar with juoyter notebook rather than vscode. It is not a problem. I make a file where you can create, train and speak to a rasa chatbot. So to do it you will need to create a jupyter notebook. It really easy ! You can attach one volume if you want to save the model created. But don't forget to put the model inside the container before stop your notebook. Here is the command to run : 

```bash
ovhai notebook run conda jupyterlab \
--name <name-notebook> \
--framework-version conda-py39-cuda11.2-v22-4 \
--volume <model-output-container>@GRA/:/workspace/trained-models:RW \
--volume https://github.com/Victor2103/rasa_chatbot.git:/workspace/public-repo-git:RO \
--cpu 10 \
--token <token> \
```

It will take one or two minutes to create your notebook and then you will be able to run the jupyter notebook. This notebook is located in the folder jupyter inside the public repository git. At the end of the notebook, you can easily speak to the chatbot. 


# Train the model with AI Training

You can clone the repository git in a folder on your computer. Then create the docker image and push it inside your repository dockerhub or in a private manage directory directly on OVHcloud. Here are the two commands to run inside the folder rasa_bot :

```bash
docker build .  -f rasa.Dockerfile -t <yourdockerhubId>/rasa-back:latest
docker push <yourdockerhubId>/rasa-back:latest
```

Here I decided to use my docker id.

Once your docker image is created and pushed into the repo, you can directly use the ovhai command to create your training of the model. Here is the full command. The training is about 5 minutes. You can change the time of the training if you change the number of gpu or the config file for the rasa training. But if you change the config file, the model will be less precise.

```bash
ovhai job run --name rasa-chatbot \
--gpu 1 \
--volume <data-to-train-container>@GRA/data:/workspace/data:RO:cache \
--volume <model-output-container>@GRA/rasa-models:/workspace/trained-models:RW \
<yourdockerhubId>/rasa-chatbot:latest \
-- bash -c "rasa train --force --out trained-models"
```

For more explanation about the CLI command for AI Training please click on this link : [CLI Reference](https://docs.ovh.com/gb/en/publiccloud/ai/cli/overview-cli/).

Explanation here for the command inside the dockerfile. 
- rasa train : This command will start to train a model with the nlu data you provide. The training launch some component and follow a pipeline defined in your file config.yml. 
- --force : This line is an option for the rasa train command. It permits to force rasa to train a model and not only search if the data provided as been already train. This option will retrain a model even if the data hasn't changed. 
- --out : This argument permits to say how you want to save your model. Here we saved the model in the folder trained-models and in the container at the mounted prefix "rasa-models". 

# Test your model

You can test shortly your model if you want. To do this, simply tap the command rasa shell in a terminal at the root of the rasa_bot folder. You will see your model loading and after, you will be able to speak with the chatbot. To have all of the functionnalities of your model, tap in another terminal at the root of the rasa_bot folder, the command : rasa run actions. Before running this two commands, don't forget to upload your rasa model in the object storage <model-output-container> into a folder name models inside the rasa_bot folder. If you forget to upload it, a model will be present already but not the model you trained.

Once you've run this 2 commands, you can speak to your chatbot ! Try to say hi or to ask about your electric consommation. He will be able to answer your questions. Maybe He can't tell you your consummation but you can have a small discussion. 

If you're not satisfied about the model because your chatbot doesn't respond very well, you can run this command. It will run again the job again and create a new model.

```bash
ovhai job rerun <job_id> 
```

To get the job id of the previous job, just run this command to get the list of the job you've run before. 

```bash
ovhai job ls
```

More explanation are here : [CLI Reference](https://docs.ovh.com/gb/en/publiccloud/ai/cli/overview-cli/).

Once you have your model is ready, we must deploy the model to use it. This will be ensure with the tool AI Deploy from the public cloud.

# Test it locally (optional)

You want to use your chatbot locally. You can do it because there is a docker compose in the project and to run the chatbot on your local machine you will just have to run one command ! But before running the command, you must go inside one dockerfile to specify directly the command to run. Let's do this. In the file rasa.Dockerfile inside the rasa_bot folder uncomment the last line. Then at the root of the git repository run : 

```bash
docker compose -f "docker-compose.yml" up -d --build
```

This command will create 3 containers, one for the rasa model, one for the rasa custom actions and one for the front end server handled by django. Once the three containers are running (it will take 5 minutes max), you can go directly on your [localhost](http://0.0.0.0:8000/) on port 8000, the port of your front end app. 

Once you've finished testing the rasa model, don't forget to stop the containers and comment the last line of the rasa.Dockerfile. To stop the containers, run this command : 

```bash
docker compose down
```

# Deploy your chatbot

For simplicity, we will use the ovhai control command again. And with one command, you will have your model running securily on a https link !

The container use for deploying your chatbot is the same as the one we use for train our chatbot. Let's run : 

```bash
ovhai app run --name rasa-back \
--unsecure-http \
--default-http-port 5005 \
--cpu 4 \
--volume <model-output-container>@GRA/rasa-models:/workspace/trained-models:RO \
<yourdockerhubId>/rasa-chatbot:latest \
-- bash -c 'rasa run -m trained-models --cors "*" --debug --connector socketio --credentials "crendentials.yml" --endpoints "endpoints.yml" & rasa run actions'
```

Now, you can wait that your app is started. Once she is started, you can go on the url and.. nothing special will append just a small message with **hello from Rasa 3.2.0** ! In fact to speak with the chatbot, we need to launch another server, a front end server. To do this, I will use the framework Django. Don't worry, everything is on the git if you clone the repository. 

# Create a Front End App

## Create your environnements variable 

So to launch the front end app, you will have to create two environments variables. The first one will be the secret key for the json web token signature to access your rasa chatbot. The second one will be the secret key to run the django application. 

To create the secret key for django app you can follow this tutorial : [Django Secret Key](https://humberto.io/blog/tldr-generate-django-secret-key/) and for the json web token key, generate a strong password with 30 characters minimal. Don't use the extended ASCII or the key will be wrong. Lots of website generate some strings but you don't know if they keep your string. It is better to use a local app like keepass or a package in python. 

You have you two environnements variables. Time to save it ! create a '.env' file inside the folder django_app/django_app. Your .env should look like this : 

```
SECRET_KEY=your-django-secret-key-generated-before
JWT_SECRET_KEY=your-jwebtoken-generated-before
JWT_ALGORITHM=HS256
```

## Create the dockerfile

Let's now run the app on AI Deploy ! To do so, you will need to create a new dockerfile. Go on the folder django_app and run simply : 

```bash
docker build . -f django.Dockerfile -t <yourdockerhubId>/front-end-chatbot:latest
docker push <yourdockerhubId>/front-end-chatbot:latest
```

## Create the app

Now let's finish this tutorial and run the front end application with the ovhai CLI. But just before, get the url of your back end rasa chatbot. It will be something like this : **https://259b36ff-fc61-46a5-9a25-8d9a7b9f8ff6.app.gra.training.ai.cloud.ovh.net/**. You can have it with the cli by listing all of your app and get the one you want. We will call this URL "RasaURL". 

Now you can run this command : 

```bash
ovhai app run --name rasa-front \
--token <token> \
--default-http-port 8000 \
-e API_URL=<RasaURL> \
--cpu 2 \
<yourdockerhubId>/front-end-chatbot:latest \
```

That's it ! On the URL of this app, you can speak to your chatbot ! Try to have a simple conversation ! And if you reload the page, you can notice that the chatbot go back to zero. So every user is different on each machine. 

If you don't use the app, don't forget to stop it. 


# Stop your app when unused

When you finish to use your model, don’t forget to stop the app. To do this simply run in the terminal : 

```bash
ovhai app stop <appid>
```

If you want to restart the rasa api, simply run again in a terminal : 

```bash
ovhai app start --token <token> \
<appid>
```