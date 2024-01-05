#!/usr/bin/env bash
set -e

# Define Variables
FOLDERS_TO_CREATE=("${HOME}/.aws/sso" "${HOME}/.polyaxon" "${HOME}/.ssh")
FILES_TO_CREATE=("${HOME}/.aws/config" "${HOME}/.gitconfig" "${HOME}/.ssh/known_hosts")

# Create folders
for folder in "${FOLDERS_TO_CREATE[@]}"; do
    echo "Creating folder $folder"
    mkdir -p $folder
done

# Create files
for file in "${FILES_TO_CREATE[@]}"; do
    echo "Creating file $file"
    touch $file
done

# Check if the expected profile is defined in the AWS config.
PYMARVEL_AWS_PROFILE="rd-thor"
PROFILE_CONFIG="[profile ${PYMARVEL_AWS_PROFILE}]\ncredential_source = Ec2InstanceMetadata\nregion = us-east-1"
if grep "profile ${PYMARVEL_AWS_PROFILE}" ~/.aws/config > /dev/null 2> /dev/null; then
  echo "AWS configuration for profile ${PYMARVEL_AWS_PROFILE} found"
else
  # If we are on EC2 instances we can inject the required configuration.
  unset AWS_PROFILE
  if aws sts get-caller-identity --query Arn | grep '/Devbox/i-' > /dev/null 2> /dev/null; then
    echo "No ${PYMARVEL_AWS_PROFILE} found on Devbox, creating one"
    echo -e "${PROFILE_CONFIG}" >> ~/.aws/config
    export AWS_PROFILE=${PYMARVEL_AWS_PROFILE}
  else

    # Warn of the missing profile but don't fail.
    echo "!!! Expected profile ${PYMARVEL_AWS_PROFILE} not configured !!!"
  fi
fi
