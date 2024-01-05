#!/usr/bin/env bash

echo "Checking AWS SSO token using AWS_PROFILE=${AWS_PROFILE}"
SSO_ACCOUNT=$(aws sts get-caller-identity --query "Account")
if [ -n "${SSO_ACCOUNT}" ]; then
    echo "AWS session still valid"
else
    echo "Seems like session expired, refreshing ..."
    if aws --version | grep 'aws-cli/1\.' >/dev/null 2>/dev/null; then
        echo "Version 1 of the AWS CLI is not supported, please upgrade to v2." >&2
        if grep -A7 "profile $AWS_PROFILE" ~/.aws/config | grep "sso_start_url" > /dev/null; then
            echo "Or if you are on a devbox, you can use the aws cli v1 profile provided in the `Prerequisites` section of the Py-Marvel documentation: https://github.com/hudl/py-marvel" >&2
        fi
        exit 1
    fi
    aws sso login
fi
