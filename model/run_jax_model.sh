#!/bin/bash

set -euo pipefail

source vae_env/bin/activate

python model/jax_planning.py "$@"

deactivate
