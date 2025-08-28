#!/bin/bash

# loading .env file
set -o allexport
source .env
set +o allexport

# login into docker hub
echo "Logging in to Docker Hub..."
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

# Build docker images from MLflow models
echo "Building the MLflow model..."
mlflow models build-docker --name "${DOCKER_IMAGE_NAME}" --model-uri "models:/${MODEL_NAME}/${MODEL_VERSION}"

echo "Pushing ${IMAGE} to Docker Hub..."
docker push "${DOCKER_USERNAME}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION}"
