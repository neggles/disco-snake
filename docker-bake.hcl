group "default" {
  targets = ["pgvector"]
}

variable "IMAGE_REGISTRY" {
  default = "ghcr.io"
}

variable "REPO_NAME" {
  default = "neggles/disco-snake"
}

variable "IMAGE_NAME" {
  default = "pgvector"
}

variable "PG_BASE_TAG" {
  default = "15.3-bullseye"
}

variable "PGVECTOR_REPO" {
  default = "https://github.com/pgvector/pgvector.git"
}

variable "PGVECTOR_REF" {
  default = "v0.4.4"
}

function "get_major" {
  params = ["version"]
  result = regex_replace(version, "([0-9]+)\\..*", "\\1")
}

# docker-metadata-action will populate this in GitHub Actions
target "docker-metadata-action" {}

# tags go in here so we can let docker-metadata-action override in GitHub Actions
target "common" {
  dockerfile = "Dockerfile"
  tags = [
    "${IMAGE_REGISTRY}/${REPO_NAME}/${IMAGE_NAME}:${PG_BASE_TAG}"
  ]
  platforms = ["linux/amd64"]
}

target "pgvector" {
  inherits = ["common", "docker-metadata-action"]
  context  = "./docker/db"
  args = {
    PG_BASE_TAG   = "${PG_BASE_TAG}"
    PG_MAJOR      = "${get_major(PG_BASE_TAG)}"
    PGVECTOR_REPO = "${PGVECTOR_REPO}"
    PGVECTOR_REF  = "${PGVECTOR_REF}"
  }
}
