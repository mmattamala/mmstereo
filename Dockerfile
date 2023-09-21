ARG BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM ${BASE_IMAGE}

# Other args
ARG BASE_REPO=https://github.com/mmattamala/mmstereo
ENV BASE_REPO=${BASE_REPO}

# Labels
LABEL maintainer="Matias Mattamala"
LABEL contact="matias@robots.ox.ac.uk"
LABEL description="Docker image to train and test mmstereo"
LABEL example_usage="docker run -it --rm --net=host --runtime nvidia mmattamala:mmstereo-pytorch2.0"

# Terminal color
SHELL ["/bin/bash", "--login", "-c"]
ENV TERM=xterm-256color

# To avoid tzdata asking for geographic location...
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_frontend noninteractive

# ==
# Installation
# ==

# Install basic packages
RUN apt update -y && apt install git -y

# Clone repo
RUN cd /workspace/ && git clone ${BASE_REPO}
RUN cd /workspace/mmstereo/ && pip install -e mmstereo/

# # Install requirements
RUN pip install virtualenv
RUN pip install -r /workspace/mmstereo/requirements.txt

# ==
# Remove cache and extra files
# ==
RUN rm -rf /var/lib/apt/lists/* && apt-get clean

CMD ["bash"]
WORKDIR /workspace/