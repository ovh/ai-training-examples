FROM --platform=linux/x86_64 python:3.8

WORKDIR /workspace
ADD requirements.txt /workspace

RUN apt-get update && apt-get install libgl1 -y
RUN pip install -r requirements.txt

ADD train-rock-paper-scissors.py /workspace

# Mandatory to run the jobs in rootless mode
RUN chown -R 42420:42420 /workspace

CMD [ "python3"  , "/workspace/train-rock-paper-scissors.py"]