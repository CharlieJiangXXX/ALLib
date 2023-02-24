#!/bin/bash

# Reorganizing Normal tiles into folders; Run in directory with files

cd ./Processed/Normal/
for i in {246..253}
do
   mkdir "${i}_M_HE"
   mv *${i}* "${i}_M_HE"
done
