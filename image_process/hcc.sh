#!/bin/bash

# Reorganizing HCC tiles into folders; Run in directory with files

cd ./Processed/HE/
for file in ../../Rat_HCC_HE/*
do
    name=$(basename "$file" .ndpi)
    echo "$name"
    mkdir "$name"
    mv *"${name}"* "$name"
done
