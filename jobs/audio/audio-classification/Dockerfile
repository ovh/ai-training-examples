FROM --platform=linux/x86_64 python:3.8

WORKDIR /workspace
ADD . /workspace

RUN pip install -r requirements.txt

# Mandatory to run the jobs in rootless mode
RUN chown -R 42420:42420 /workspace

CMD [ "python3"  , "/workspace/train-audio-classification.py"]