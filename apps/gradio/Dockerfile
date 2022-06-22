FROM python:3.7

WORKDIR /workspace
ADD requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

ADD app.py mnist-classes.png /workspace/

RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
CMD [ "python3" , "/workspace/app.py" ]
