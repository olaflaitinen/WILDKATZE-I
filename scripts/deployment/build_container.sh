#!/bin/bash
set -e

IMAGE_NAME=${1:-wildkatze-api}
TAG=${2:-latest}

echo "Building Docker image: $IMAGE_NAME:$TAG"

docker build -f docker/Dockerfile.api -t $IMAGE_NAME:$TAG .

echo "Build complete."
