FROM ovhcom/ai-training-one-for-all

WORKDIR /workspace
ADD . /workspace

RUN pip install -r requirements.txt

CMD [ "streamlit" , "run" , "/workspace/main.py", "--server.address=0.0.0.0" ]

RUN mkdir /data ; chown -R 42420:42420 /workspace /data

ENV HOME=/workspace
