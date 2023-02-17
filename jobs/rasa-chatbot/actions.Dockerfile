# This dockerfile is only needed if we will use a docker-compose file. 
FROM python:3.8

WORKDIR /workspace
ADD actions /workspace/
COPY requirements_rasa.txt /workspace/

RUN pip install --no-cache-dir -r requirements_rasa.txt


RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace


#If you deploy the chatbot you expose at port 5005.
EXPOSE 5055

# Command to run for the actions
#CMD rasa run actions
