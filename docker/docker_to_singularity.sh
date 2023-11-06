#!/bin/bash

USER=mmattamala
IMAGE=mmstereo
TAG=pytorch2.0

DOCKER_IMAGE=${USER}/${IMAGE}:${TAG}
SINGULARITY_IMAGE=${USER}-${IMAGE}-${TAG}

SINGULARITY_NOHTTPS=1 singularity build --sandbox ${SINGULARITY_IMAGE}.sif docker-daemon://${DOCKER_IMAGE}
