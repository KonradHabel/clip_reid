SUMO_URL=$(${AWS_CLI_PATH}aws secretsmanager get-secret-value --secret-id DevcontainerSumoCollectorUrl --query SecretString --output text)
DEVCONTAINER_NAME=$(grep -o '"name":.*' .devcontainer/devcontainer.json | cut -d '"' -f 4)
BUILD_ID_LOCATION=".pymarvel/build_id.txt"
STATE=$1

if [ "$STATE" == "STARTED" ] && [ ! -f $BUILD_ID_LOCATION ]; then
    openssl rand -hex 12 > $BUILD_ID_LOCATION
fi

BUILD_ID=$(cat $BUILD_ID_LOCATION)

curl -X POST -H "Content-Type:application/json" -H "X-Sumo-Category:app_aml_devcontainer" \
    -H "X-Sumo-Name:$DEVCONTAINER_NAME" \
    -d '{"message": "AML devcontainer state update", "state": "'"$1"'", "id": "'"$BUILD_ID"'", "user": "'"$(whoami)"'"}' \
    "$SUMO_URL"  

if [ "$STATE" == "SUCCEEDED" ]; then
    rm $BUILD_ID_LOCATION
fi
