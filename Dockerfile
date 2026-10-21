FROM ubuntu:22.04

# procps provides ps, which Nextflow needs for task metrics.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        git \
        openjdk-11-jdk-headless \
        procps \
        python3-dev \
        python3-pip \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Pinned with: uv pip compile (setup.py requires + numcodecs)
#   --python-version 3.10 --python-platform x86_64-manylinux_2_28
COPY docker/constraints.txt /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

RUN pip3 install -q -U pip && pip3 install -q numpy numcodecs

COPY / /app/ashlar/
RUN pip3 install /app/ashlar && ashlar --version

ENV OMP_NUM_THREADS=1
# Matplotlib needs a writable config dir when run as an arbitrary user.
ENV MPLCONFIGDIR=/tmp/matplotlib

VOLUME /data
VOLUME /data2
VOLUME /data3

WORKDIR /data
