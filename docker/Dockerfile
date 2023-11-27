ARG BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM ${BASE_IMAGE}

# Other args
ARG BASE_REPO=https://github.com/mmattamala/mmstereo
ENV BASE_REPO=${BASE_REPO}

# Labels
LABEL maintainer="Matias Mattamala"
LABEL contact="matias@robots.ox.ac.uk"
LABEL description="Docker image to train and test mmstereo"
LABEL example_usage="docker run -it --rm --net=host --runtime nvidia mmattamala/mmstereo:pytorch2.0"

ARG ROS_VERSION="noetic"

# Terminal color
SHELL ["/bin/bash", "--login", "-c"]
ENV TERM=xterm-256color

# To avoid tzdata asking for geographic location...
ENV DEBIAN_FRONTEND=noninteractive

# ==
# Installation
# ==

# ==
# Install ROS
# ==
RUN apt update \
    && apt install --no-install-recommends -y curl gnupg lsb-release git build-essential nano \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt update \
    && apt install --no-install-recommends -y ros-${ROS_VERSION}-desktop-full \
    && apt install --no-install-recommends -y \
        python3-pip \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-catkin-tools \
        python3-osrf-pycommon \
    && rm -f "/etc/ros/rosdep/sources.list.d/20-default.list" \
    && rosdep init \
    && rosdep update \
    && echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc



# Install basic packages
RUN apt update -y && apt install -y --no-install-recommends \
    git libturbojpeg libgl1 libglib2.0-0

# # Clone repo
# RUN cd /workspace/ && git clone ${BASE_REPO}
# RUN cd /workspace/mmstereo/ && pip install -e mmstereo/

# # Install requirements
COPY ../requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt && rm /workspace/requirements.txt

# ==
# Remove cache and extra files
# ==
RUN rm -rf /var/lib/apt/lists/* && apt-get clean

ENV DEBIAN_FRONTEND=dialog
CMD ["bash"]
WORKDIR /root/