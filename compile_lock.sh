#!/bin/bash

base_image="python:3.11-slim-bullseye"
docker pull \
    "${base_image}"

docker run --rm \
    -v $PWD/binder:/read \
    "${base_image}" /bin/bash -c 'python -m venv venv && \
    . venv/bin/activate && \
    command -v python &&
    apt-get update && \
    apt-get install --no-install-recommends -yq git && \
    python -m pip install --upgrade \
        pip \
        setuptools \
        wheel && \
    python -m pip install pip-tools && \
    cp /read/requirements.in . && \
    pip-compile --generate-hashes --output-file=requirements.txt --resolver=backtracking requirements.in && \
    cp requirements.txt /read/'

# Make lockfile local user owned
mv binder/requirements.txt tmp.lock
cp tmp.lock binder/requirements.txt
rm tmp.lock
