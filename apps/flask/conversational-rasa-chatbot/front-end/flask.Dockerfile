FROM python:3.8

WORKDIR /workspace
ADD . /workspace

RUN pip install --no-cache-dir -r requirements_flask.txt


RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace



EXPOSE 5000

CMD python3 app.py