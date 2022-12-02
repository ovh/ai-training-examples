## Speech to text Application using *Python* & *Streamlit* and pre-trained models

This speech to text application is based on 2 [notebook tutorials](https://github.com/ovh/ai-training-examples/tree/main/notebooks/natural-language-processing/speech-to-text/conda) *(Basics and advanced parts)*. To better understand the code, we recommend that you read these notebooks.

### Files description
- `requirements.txt` and `packages.txt` files contain the librairies used by our speech to text app
- `main.py` and `app.py` are the files of the application
- `Dockerfile` allows you to deploy your app on **AI Deploy**
- `download_models.py` is an optional script that allows you to download the models required by our Speech to Text Application and save them in a folder or an Object Storage with AI Deploy.
Storing your models in a folder will save you some time. Indeed, you will not have to download them every time you launch the app!

### Important
To make it work with the diarization option (speakers differentiation), do not forget to **replace the default token by your own one** in the following line of code of the `app.py` file:

<code>
dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="ACCESS TOKEN GOES HERE")
</code>


