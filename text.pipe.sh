#!/bin/bash

CONFIG="./config/dictionary-def.yml"
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

#timeit "uv run manage.py get-embeddings --config $CONFIG" "get-embeddings"
#timeit "uv run manage.py encoder test --config $CONFIG --n 100" "encoder test"
timeit "uv run manage.py get-bestparams --config $CONFIG" "get-bestparams"

CONFIG="${CONFIG/.yml/.best.yml}"

#timeit "uv run manage.py plot --config $CONFIG" "plot"
#timeit "uv run manage.py create-memories --config $CONFIG --n 1000" "create-memories"