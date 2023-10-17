#!/usr/bin/env bash

#!/usr/bin/env bash

echo "Undeploying..."

if kubectl kustomize . | kubectl delete -f - --ignore-not-found=true; then
    echo "Udeployed!"
else
    echo "Undeployment failed!"
    exit 1
fi
