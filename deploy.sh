#!/usr/bin/env bash

echo "Deploying..."

if kubectl kustomize . | kubectl apply -f -; then
    echo "Deployed!"
else
    echo "Deployment failed!"
    exit 1
fi
