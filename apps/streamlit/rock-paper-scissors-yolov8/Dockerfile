# 🐳 Base image to execute Python application
FROM python:3.8

# 👱 Userspace for AI Deploy
WORKDIR /workspace
# 📚 Libraries for the application
ADD requirements.txt /workspace
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# 🐍 Python application
ADD app.py /workspace

# 👮 Rootless execution
RUN chown -R 42420:42420 /workspace

# 🚀 Run the Python application
CMD [ "streamlit" , "run" , "/workspace/app.py", "--server.address=0.0.0.0" ]

ENV HOME=/workspace