#!/bin/bash

CONFIG="./config/SPOTS-256.yml"

set -e

uv run manage.py clean-logs
uv run manage.py autoencoder train --config $CONFIG
uv run manage.py get-embeddings --config $CONFIG
uv run manage.py autoencoder test --config $CONFIG
uv run manage.py classifier train --config $CONFIG
uv run manage.py classifier test --config $CONFIG
uv run manage.py create-memories --config $CONFIG