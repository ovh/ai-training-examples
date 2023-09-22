# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
# Copy the application code and requirements file into the container
WORKDIR /workspace
ADD . /workspace

# Install the Python dependencies
RUN pip install -r requirements.txt

# Give correct access rights to the OVHcloud user
RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace

# Define the command to run the Streamlit application when the container is launched
CMD [ "streamlit" , "run" , "/workspace/main.py", "--server.address=0.0.0.0" ]

