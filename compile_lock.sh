#!/bin/bash

base_image="python:3.11-slim-bullseye"
docker pull \
    "${base_image}"

docker run --rm \
    -v $PWD/binder:/read \
    python -m venv venv && \
    . venv/bin/activate && \
    command -v python &&
    python -m pip install --upgrade \
        pip \
        setuptools \
        wheel && \
    python -m pip install pip-tools && \
    cp /read/requirements.txt . && \
    pip-compile --generate-hashes --output-file requirements.lock requirements.txt && \
    cp requirements.lock /read/'

# Make lockfile local user owned
mv requirements.lock tmp.lock
cp tmp.lock requirements.lock
rm tmp.lock