FROM python:3.12-slim-bookworm AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    coreutils findutils grep sed gawk jq curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/work
WORKDIR /work
USER 65534:65534
CMD ["/bin/sh"]

FROM base AS selfmod
USER root
COPY pyproject.toml /opt/project/pyproject.toml
COPY nezubot_discrod /opt/project/nezubot_discrod
ENV PATH="/opt/venv/bin:${PATH}" PYTHONPATH=/work
RUN python -m venv /opt/venv && pip install --no-cache-dir '/opt/project[dev]' \
    && pip uninstall -y nezubot-discrod && rm -rf /opt/project
USER 65534:65534

FROM base AS sandbox
