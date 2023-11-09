FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /workspace
ADD . /workspace

RUN pip install --no-cache-dir -r requirements_rasa.txt


RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace


#If you deploy the chatbot you expose at port 5005.
EXPOSE 5005 


CMD rasa train --force --out trained-models
