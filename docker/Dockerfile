FROM python:3.12-slim

LABEL authors="giacomo.fabrini@crick.ac.uk" \
      description="UDE Training Pipeline Image"

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
