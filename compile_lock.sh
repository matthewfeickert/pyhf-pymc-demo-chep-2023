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
    cp /read/requirements.txt . && \
    pip-compile --generate-hashes --output-file=requirements.lock --resolver=backtracking requirements.txt && \
    cp requirements.lock /read/'

# Make lockfile local user owned
mv binder/requirements.lock tmp.lock
cp tmp.lock binder/requirements.lock
rm tmp.lock
