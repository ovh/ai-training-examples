FROM python:3.8

WORKDIR /workspace
ADD . /workspace

RUN pip install --no-cache-dir -r requirements_rasa.txt


RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace


#If you deploy the chatbot you expose at port 5005.
EXPOSE 5005 


CMD rasa run -m trained-models --cors "*" --debug --connector socketio --credentials "crendentials.yml" --endpoints "endpoints.yml" & rasa run actions
