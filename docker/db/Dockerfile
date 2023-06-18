# syntax=docker/dockerfile:1

ARG DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_PRIORITY=critical

ARG PG_BASE_TAG=15.3-bullseye
ARG PG_MAJOR=15
ARG PGVECTOR_REPO=https://github.com/pgvector/pgvector.git
ARG PGVECTOR_REF=v0.4.4

FROM postgres:${PG_BASE_TAG} AS builder

# Set up the shell
SHELL [ "/bin/bash", "-ceuxo", "pipefail" ]

# set up some important environment variables
ARG DEBIAN_FRONTEND
ARG DEBIAN_PRIORITY

ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Install build dependencies
ARG PG_MAJOR
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update \
  && apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    postgresql-server-dev-$PG_MAJOR \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Set work dir and clone pgvector
WORKDIR /work
ARG PGVECTOR_REPO
ARG PGVECTOR_REF
RUN git clone --depth=1 "${PGVECTOR_REPO}" "pgvector" \
  && cd pgvector \
  && git fetch --depth=1 origin "${PGVECTOR_REF}" \
  && git reset --hard FETCH_HEAD \
  && rm -fr .git

# Build pgvector
ARG OPTFLAGS=""
RUN cd /work/pgvector \
    && make clean \
    && make -j2 OPTFLAGS="${OPTFLAGS}" DESTDIR="/work/pgvector/dist" \
    && make DESTDIR="/work/pgvector/dist" install

# Copy pgvector to final image, so we don't waste space on build deps etc.
FROM postgres:${PG_BASE_TAG} AS pgvector

# Set up the shell
SHELL [ "/bin/bash", "-ceuxo", "pipefail" ]

# set up some important environment variables
ARG DEBIAN_FRONTEND
ARG DEBIAN_PRIORITY

ENV LC_ALL=C.UTF-8

# copy pgvector from builder
COPY --from=builder /work/pgvector/dist /pgvector

# Copy files into place and clean up
RUN cp -a /pgvector/usr/* /usr/ \
  && rm -fr /pgvector
