#!/bin/bash

for i in {1..100}; do
    target="${i}_0"
    rm -rf "$target"
    #find . -type d -name "$target" -exec rm -rf {} +
done