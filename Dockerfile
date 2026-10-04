FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/immunotargetnet/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/immunotargetnet

RUN apt-get update \
    && apt-get install --no-install-recommends --yes build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY configs ./configs
COPY assets ./assets
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "immunotargetnet.cli.evaluate", "--epochs", "2", "--seeds", "1"]
