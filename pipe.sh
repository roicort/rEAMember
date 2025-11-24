#!/bin/bash

CONFIG="./config/fashion.yml"
TIMES_FILE="times.txt"

set -e

echo "" > $TIMES_FILE

function timeit() {
    CMD="$1"
    LABEL="$2"
    START=$(date +%s)
    eval "$CMD"
    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "$LABEL: $ELAPSED seconds" >> $TIMES_FILE
}

timeit "uv run manage.py clean-logs" "clean-logs"
timeit "uv run manage.py encoder train --config $CONFIG" "encoder train"
timeit "uv run manage.py get-embeddings --config $CONFIG" "get-embeddings"
timeit "uv run manage.py encoder test --config $CONFIG --n 100" "encoder test"
timeit "uv run manage.py classifier train --config $CONFIG" "classifier train"
timeit "uv run manage.py classifier test --config $CONFIG" "classifier test"
timeit "uv run manage.py get-bestparams --config $CONFIG" "get-bestparams"

CONFIG="${CONFIG/.yml/.best.yml}"

timeit "uv run manage.py plot --config $CONFIG" "plot"
timeit "uv run manage.py create-memories --config $CONFIG --n 100" "create-memories"
