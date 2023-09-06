FROM python:3.10

WORKDIR /workspace
ADD . /workspace

RUN pip install -r requirements.txt

CMD [ "python3"  , "/workspace/train_image_classification.py"]

RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
