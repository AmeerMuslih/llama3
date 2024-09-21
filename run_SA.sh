#!/bin/bash

for dir in ~/Ameer/llama3/Matrices/*/; do
    if [ -d "$dir" ]; then
        echo "$(basename "$dir")"
    fi
done