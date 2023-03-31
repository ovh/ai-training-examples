# ASL recognition using YOLOv7

*The previous code will be used to launch a notebook and an app from **OVHcloud AI Tools**.*

📄 Access to Weights & Biases report [here](https://api.wandb.ai/links/asl-alphabet-data-augment-ovh/wau6k1rd).

## Requirements

- a Public Cloud project in your OVHcloud account
- access to the OVHcloud Control Panel
- `ovhai` CLI installed
- a Public Cloud user with `administrator` or `AI Training admin` role, see here for more information

*You will find all information on OVHcloud [documentation](https://docs.ovh.com/gb/en/publiccloud/ai/).*

## Create Object Storage container 

- Create the **data** container (empty): `ovhai data upload gra data-sign-language`

- Create the **model** container (empty): `ovhai data upload gra model-sign-language`

- Create the **images** container (with your own images to do detection on your future model): `ovhai data upload gra images-sign-language repo-local-my-test-images/ --remove-prefix repo-local-my-test-imagesl/`

## Launch an AI Notebook

To launch and access to the AI Notebook, you have to launch the following command:

```
ovhai notebook run miniconda jupyterlab \
	--name notebook-yolov7-asl \
	--framework-version conda-py39-cuda11.2-v22-4 \
	--gpu 1 \
	--volume data-sign-language@GRA/:/workspace/data:RW:cache \
	--volume model-sign-language@GRA/:/workspace/asl-yolov7-model:RW \
	--volume images-sign-language@GRA/:/workspace/images:RO \
	--volume https://github.com/eleapttn/yolov7_streamlit_asl_recognition.git:/workspace/github-repo:RW
```

## Launch an AI Deploy app

To launch and access to the AI Deploy app, you have to launch the following command:

> First, you have to build and push your Docker image to your Docker Hub!

```
ovhai app run <your_docker_id>/yolov7-asl-recognition:latest \
	--gpu 1 \
	--default-http-port 8501 \
	--volume asl-volov7-model@GRA/:/workspace/asl-volov7-model:RO
```

## References 

Access to the resources:

- Slides are available [here](https://noti.st/eleapttn/ZuK1ot/what-if-ai-was-the-solution-to-understand-sign-language).
- You can check the replay on this [link](https://summit2022.aiforhealth.fr/onlinesession/5c6f487b-9252-ed11-819a-000d3a45cc82).
