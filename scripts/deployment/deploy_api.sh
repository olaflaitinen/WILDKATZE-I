#!/bin/bash
set -e

NAMESPACE=${1:-wildkatze-prod}

echo "Deploying WILDKATZE-I API to Kubernetes..."

kubectl apply -f kubernetes/configmap.yaml -n $NAMESPACE
kubectl apply -f kubernetes/deployment.yaml -n $NAMESPACE
kubectl apply -f kubernetes/service.yaml -n $NAMESPACE
kubectl apply -f kubernetes/ingress.yaml -n $NAMESPACE

echo "Deployment complete."
