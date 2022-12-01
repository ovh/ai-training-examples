FROM python:3.8

WORKDIR /workspace
ADD . /workspace

RUN apt-get update
RUN xargs -a packages.txt apt-get install --yes
RUN pip install -r requirements.txt

CMD [ "streamlit" , "run" , "/workspace/main.py", "--server.address=0.0.0.0" ]

RUN mkdir /data ; chown -R 42420:42420 /workspace /data

ENV HOME=/workspace
