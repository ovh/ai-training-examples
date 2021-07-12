FROM pytorch/pytorch

WORKDIR /workspace

ADD . /workspace

RUN pip install -r requirements.txt

CMD [ "python" , "/workspace/app.py" , "--host=0.0.0.0" ]

RUN chown -R 42420:42420 /workspace

ENV HOME=/workspace
