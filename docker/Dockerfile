# Stage 1: Build dependencies
FROM python:3.12-slim as builder

# Set /usr/wheels as working directory for pre-build
WORKDIR /usr/wheels

# Install build dependencies for fully functional requirements.txt install
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes \
    gcc \
    g++ && \
#    git \
#    python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip-tools to compile requirements and build wheels
RUN pip install --no-cache-dir pip-tools wheel

# Copy requirements files
COPY requirements.txt requirements.txt

# Compile and download wheels
RUN pip wheel --wheel-dir /usr/wheels -r requirements.txt

# Stage 2: Create final image
FROM python:3.12-slim
LABEL authors="giacomo.fabrini@crick.ac.uk" \
    description="Docker image containing requirements for Test_UDE"

# Update pip to latest version
RUN python -m pip install --upgrade pip

# Set /usr/src/ as the working directory
WORKDIR /usr/src/

# Copy everything from /src to the working directory
COPY src/ .

# Copy wheels from builder stage
COPY --from=builder /usr/wheels /usr/wheels

# Install wheels
RUN pip install --no-cache-dir --no-index --find-links=/usr/wheels -r /usr/wheels/requirements.txt

# Install project modules
RUN python -m pip install .

# Set PYTHONPATH to include the source directory
ENV PYTHONPATH="/usr/src/:$PYTHONPATH"

ENTRYPOINT ["/bin/bash"]