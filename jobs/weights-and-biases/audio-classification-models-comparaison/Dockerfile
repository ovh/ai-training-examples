FROM python:3.9

WORKDIR /workspace
ADD . /workspace
 
RUN apt-get update
RUN xargs -a packages.txt apt-get install --yes
RUN pip install --no-cache-dir -r requirements.txt

RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
