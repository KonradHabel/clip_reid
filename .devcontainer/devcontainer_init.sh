#!/usr/bin/env bash
set -e

# Script to collect all devcontainer initialisation commands.
# This keeps the devcontainer.json file cleaner and the init command more readable.
#
# This script must be run from the parent directory of the `.devcontainer` directory.
#
# --- Project customisation hook ---
# This script will invoke the optional `.devcontainer/custom_init.sh` script.
# Create this file to inject project specific steps without clashing with tempalted files.
# NOTE: the custom init script must be executable or it will be ignored.

export AWS_PROFILE=rd-thor
.pymarvel/prepare_mounts.sh
.pymarvel/aws_sso_login.sh
.pymarvel/log_devcontainer_event.sh STARTED
.pymarvel/build_tokens.sh docker

# Inject optional project customisation at the end.
PY_MARVEL_CUSTOM_INIT=".devcontainer/custom_init.sh"
if [ -x "${PY_MARVEL_CUSTOM_INIT}" ]; then
  echo "Applying project init customisations"
  "${PY_MARVEL_CUSTOM_INIT}"
fi
