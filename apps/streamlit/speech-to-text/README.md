## Speech to text Application using *Python* & *Streamlit* and pre-trained models

This speech to text application is based on 2 [notebook tutorials](https://github.com/ovh/ai-training-examples/tree/main/notebooks/natural-language-processing/speech-to-text/conda) *(Basics and advanced parts)*. To better understand the code, we recommend that you read these notebooks.

To make it work with the diarization option (speakers differentiation), do not forget to replace the default token by your own one in the following line of code:

<code>
dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="ACCESS TOKEN GOES HERE")
</code>
