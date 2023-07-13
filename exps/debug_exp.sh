#!/bin/bash

# RUN_LOCAL ensures we run the code from here, not to the copied experiment
# directory.
RUN_LOCAL=1 bash main.sh sbgm "$@"
