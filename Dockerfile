# Environment for the practical ML course: Python 3.8 and the exact library versions the course was tested with.
# The course files are NOT copied into the image: docker-compose.yml mounts the project folder, so your work is saved on your computer.
FROM python:3.8-slim

# libgomp1: the OpenMP runtime that XGBoost and scikit-learn need on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /course

COPY requirements-lock.txt /tmp/requirements-lock.txt
# the lock file also pins JupyterLab 3.6, the last release line that supports Python 3.8
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements-lock.txt

EXPOSE 8888

# Default command: open JupyterLab on the course folder. Only reachable from your own computer (see docker-compose.yml).
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password=", "--notebook-dir=/course"]
