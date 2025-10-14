#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parent directory of the script's location
NEW_PATH="$(dirname "$SCRIPT_DIR")"

# Check if NEW_PATH is already in PYTHONPATH
if [[ ":$PYTHONPATH:" != *":$NEW_PATH:"* ]]; then
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$NEW_PATH"
  echo "Added $NEW_PATH to PYTHONPATH."
else
  echo "$NEW_PATH is already in PYTHONPATH."
fi
