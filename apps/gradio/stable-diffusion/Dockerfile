# 🐳 Base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 🚫 Remove interactivity since using the base image will ask for a timezone - This allows to not provide it
ENV DEBIAN_FRONTEND=noninteractive

# 📚 Install missing system packages (git, libgl1, ..., are needed for Stable Diffusion and are not installed in the base image)
RUN apt-get update && \
    apt-get install -y wget git python3 python3-venv libgl1 libglib2.0-0

# 👱 Set the working directory inside the container
WORKDIR /workspace

# 📥 Download the webui.sh script from the specified URL
RUN wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh

# 👮‍♀️ Make the webui.sh script executable
RUN chmod +x webui.sh

# ⌛️ Install the webui.sh file (--exit parameter allows to only install it without without running it)
RUN ./webui.sh -f --exit

# 🔑 Give correct access rights to the OVHcloud user
RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace
