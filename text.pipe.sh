#!/bin/bash

CONFIG="./config/dictionary-def.yml"
TIMES_FILE="${CONFIG/.yml/.times.txt}"

set -e

echo "" > $TIMES_FILE

function timeit() {
    CMD="$1"
    LABEL="$2"
    START=$(date +%s)
    eval "$CMD"
    END=$(date +%s)
    ELAPSED=$((END - START))

    DAYS=$((ELAPSED / 86400))
    HOURS=$(((ELAPSED % 86400) / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))

    printf "%s: %dd %02dh %02dm %02ds\n" "$LABEL" "$DAYS" "$HOURS" "$MINUTES" "$SECONDS" >> "$TIMES_FILE"
}

timeit "uv run manage.py get-embeddings --config $CONFIG" "get-embeddings"
timeit "uv run manage.py encoder test --config $CONFIG --n 100" "encoder test"
timeit "uv run manage.py test-recall --config $CONFIG" "test-recall"
timeit "uv run manage.py get-bestparams --config $CONFIG" "get-bestparams"
timeit "uv run manage.py get-bestparams --config $CONFIG --noise" "get-bestparams"

CONFIG="${CONFIG/.yml/.best.yml}"
#CONFIG="${CONFIG/.yml/.best.noise.yml}"

timeit "uv run manage.py plot --config $CONFIG" "plot"
timeit "uv run manage.py create-memories --config $CONFIG --n 1000" "create-memories"
timeit "uv run manage.py create-memories --config $CONFIG --noise --n 1000" "create-memories"