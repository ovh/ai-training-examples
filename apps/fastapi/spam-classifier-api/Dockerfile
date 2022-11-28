FROM python:3.8

WORKDIR /workspace
ADD . /workspace

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["app:app", "--host", "0.0.0.0"]

RUN chown -R 42420:42420 /workspace

ENV HOME=/workspace
