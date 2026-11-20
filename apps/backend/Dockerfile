# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build-base

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv "$VIRTUAL_ENV"
COPY requirements/ requirements/
RUN pip install --requirement requirements/base.txt
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps .

FROM build-base AS build-cpu
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_FIND_LINKS
# An empty index permits verified local wheels when the PyTorch CDN is unreachable.
RUN sed '/^--extra-index-url /d' requirements/cpu.txt > requirements/cpu-build.txt \
    && if [ -n "$TORCH_CPU_INDEX_URL" ]; then \
         pip install --requirement requirements/cpu-build.txt --extra-index-url "$TORCH_CPU_INDEX_URL"; \
       else pip install --requirement requirements/cpu-build.txt; fi

FROM build-base AS build-gpu
RUN pip install --requirement requirements/gpu.txt

FROM ${PYTHON_IMAGE} AS runtime-base

ARG APP_UID=10001
ARG APP_GID=10001
ARG APP_VERSION=0.1.0

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_CONFIG_FILE=/config/app.yaml \
    APP_MIGRATIONS_ROOT=/app \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

LABEL org.opencontainers.image.title="Spina Bifida backend" \
      org.opencontainers.image.version="$APP_VERSION"

ARG DEBIAN_MIRROR=http://deb.debian.org/debian
ARG DEBIAN_SECURITY_MIRROR=http://deb.debian.org/debian-security
RUN sed -i "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g; s|http://deb.debian.org/debian\$|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "$APP_GID" app \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app /config /data/db /data/jobs /data/feedback /data/logs /data/backups /models \
    && chown -R app:app /app /data /tmp

WORKDIR /app
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations/ migrations/

USER app:app
EXPOSE 8000

FROM runtime-base AS runtime-cpu
COPY --from=build-cpu --chown=app:app /opt/venv /opt/venv
RUN python -c "import cv2; from ultralytics import YOLO"
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "spina_bifida:create_app()"]

FROM runtime-cpu AS test-cpu
USER root
COPY requirements/dev.txt /tmp/requirements/dev.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements/dev.txt \
    && rm -rf /tmp/requirements
USER app:app

FROM runtime-base AS runtime-gpu
COPY --from=build-gpu --chown=app:app /opt/venv /opt/venv
RUN python -c "import cv2; from ultralytics import YOLO"
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "spina_bifida:create_app()"]
