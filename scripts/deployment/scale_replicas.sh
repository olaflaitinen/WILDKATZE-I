#!/bin/bash
set -e

REPLICAS=${1:-3}
NAMESPACE=${2:-wildkatze-prod}

echo "Scaling WILDKATZE-I API to $REPLICAS replicas..."

kubectl scale deployment wildkatze-api --replicas=$REPLICAS -n $NAMESPACE

echo "Scaling complete."
