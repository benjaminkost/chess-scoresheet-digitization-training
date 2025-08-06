#!/bin/bash

# loading .env file
set -o allexport
source .env
set +o allexport

# login into docker hub
echo "Logging in to Docker Hub..."
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

# Build docker images from MLflow models
mlflow models build-docker --name "${IMAGE}" --model-uri "models:/${MODEL_NAME}/${MODEL_VERSION}"

echo "Pushing ${IMAGE} to Docker Hub..."
docker push "${DOCKER_USERNAME}/${IMAGE}:${DOCKER_IMAGE_VERSION}"
