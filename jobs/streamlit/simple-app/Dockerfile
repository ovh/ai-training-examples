# Choose your base image
FROM python:3.8
# Install your dependencies
RUN pip install streamlit pandas numpy
# Install your application
COPY simple_app.py /opt/simple_app.py
# Define your default launching command
CMD [ "streamlit" , "run" , "/opt/simple_app.py", "--server.address=0.0.0.0" ]
# Create a HOME dedicated to the ovhcloud user (42420:42420)
RUN mkdir /workspace && chown -R 42420:42420 /workspace
ENV HOME /workspace
WORKDIR /workspace