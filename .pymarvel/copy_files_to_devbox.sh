#!/bin/bash

# Usage: ./copy_files_to_devbox.sh <devbox_ip_address> <devbox_path>
# In case you are building a new py-marvel project and want to test it out on a devbox without committing to GitHub
# you can use this script to copy the files to the devbox. 
# Perhaps you want to test out some template changes and want to see the results with a throw-away py-marvel project.
# This script is not used in the normal py-marvel workflow.

# Variables
DEVBOX_IP_ADDRESS=$1
DEVBOX_PATH=$2

if [ -z "$DEVBOX_IP_ADDRESS" ]; then
    echo "Please provide the IP address of the devbox as the first argument"
    exit 1
fi
if [ -z "$DEVBOX_PATH" ]; then
    echo "Please provide the path on the devbox to which files should be copied as the second argument"
    exit 1
fi

USER="ec2-user"
COMPLETE_PATH="/home/$USER/$DEVBOX_PATH"
REMOTE_DESTINATION="$USER@$DEVBOX_IP_ADDRESS:$COMPLETE_PATH"

DIR="$(pwd)"
rm -r "$DIR"/.mypy_cache
rm -r "$DIR"/.pytest_cache
rm -r "$DIR"/.venv

ssh "$USER"@$DEVBOX_IP_ADDRESS "mkdir -p $COMPLETE_PATH"

# Copy folder to EC2 instance
scp -r $DIR "$REMOTE_DESTINATION"
