#!/usr/bin/env bash
set -e

# URL of the AML ECR Registry to check if the ECR Credentials helper is configured.
AWS_DOMAIN_OWNER="690616407375"
AWS_REGION="us-east-1"
ECR_DOMAIN="${AWS_DOMAIN_OWNER}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# File to store the access token into.
PYPI_TOKEN_LOCATION=${PYPI_TOKEN_LOCATION:-".devcontainer/pypi-token.txt"}

# How long tokens will be valid for (30 minutes by default).
TOKEN_TTL=${TOKEN_TTL:-1800}

## INDIVIDUAL STEPS ##

# Log into ECR, if docker is installed and ECR Credentials gelper is not.
docker_access() {
  if which docker >/dev/null 2>/dev/null; then
    if docker_ecr_ready; then
      echo "Found ECR Credentials helper, skipping ECR Login"
    else
      echo "Configuring ECR access for Docker"
      echo "AWS_PROFILE: ${AWS_PROFILE}"
      ${AWS_CLI_PATH}aws ecr get-login-password --profile $AWS_PROFILE --region $AWS_REGION | \
        docker login --username AWS --password-stdin $AWS_DOMAIN_OWNER.dkr.ecr.$AWS_REGION.amazonaws.com
    fi
  else
    echo "Docker not found, skpping login"
  fi
}

# Check if the ECR credentials helper is installed and configured.
docker_ecr_ready() {
  which docker-credential-ecr-login >/dev/null 2>/dev/null
}

# Generate an access token to the CodeArtifacts internal repository.
# This is used by both tools (such as poetry below) and docker intermediate build stages.
fetch_token() {
  echo "Fetching fresh internal PyPi token"
  ${AWS_CLI_PATH}aws codeartifact get-authorization-token \
    --domain hudlaml \
    --domain-owner 690616407375 \
    --duration-seconds "${TOKEN_TTL}" \
    --query authorizationToken \
    --output text \
    --region us-east-1 > "${PYPI_TOKEN_LOCATION}"
}

# Configure pip with additional access to the internal packages.
# This is used to pip install build tools during the INSTALL phase.
pip_access() {
  echo "Configuring PyPi access for pip"
  ${AWS_CLI_PATH}aws codeartifact login \
    --tool pip \
    --domain hudlaml \
    --domain-owner 690616407375 \
    --repository hudlaml \
    --duration-seconds "${TOKEN_TTL}"

  # We want the internal index to be extra, not the only option.
  # Since this is not supported by the aws cli we shuffle pip's config after logging in.
  pip config set 'global.extra-index-url' "$(pip config get 'global.index-url')"
  pip config unset 'global.index-url'
}

# Configure poetry, if present, in the build environment with access to internal packages.
# This is used by library builds to fetch dependencies during the INSTALL phase.
poetry_apply() {
  if which poetry >/dev/null 2>/dev/null; then
    echo "Configuring PyPi access for poetry"
    poetry config http-basic.hudl aws $(cat "${PYPI_TOKEN_LOCATION}")
  else

    echo "Poetry not found, skipping configuration"
  fi
}

## ENTRY POINT ##
case "$1" in
  "")
    fetch_token
    pip_access
    poetry_apply
    docker_access
    ;;
  fetch-token)
    fetch_token
    ;;
  pip)
    pip_access
    ;;
  poetry)
    fetch_token
    poetry_apply
    ;;
  poetry-apply)
    poetry_apply
    ;;
  python)
    fetch_token
    pip_access
    poetry_apply
    ;;
  docker)
    docker_access
    ;;
  *)
    echo "Unsupported option '${1}'" >&2
    echo "" >&2
    echo "Usage: ./build_tokens.sh [MODE]" >&2
    echo "" >&2
    echo "MODE can be one of:" >&2
    echo "  fetch-token    Fetch a PyPI token to access the AML internal repository" >&2
    echo "  pip            Configure PIP to use the AML intenral repository as a secondary source" >&2
    echo "  poetry         Shorthand for fetch-token followed by poetry-apply" >&2
    echo "  poetry-apply   Configure access to the AML internal repository for Poetry" >&2
    echo "  docker         Configure docker access to AML private repositories" >&2
    exit 1
    ;;
esac
