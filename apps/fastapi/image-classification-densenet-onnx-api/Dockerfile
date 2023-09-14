FROM python:3.10

WORKDIR /workspace
ADD . /workspace

RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn"]
CMD ["app:app", "--host", "0.0.0.0", "--port", "8080"]

RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
