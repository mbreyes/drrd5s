#!/bin/bash

PREFIX=AR7
DIR="../../data/raw/AR7"

for rat in {1..14};
do
    #ls "$DIR"/"$PREFIX"$(printf "%03d\n" "$rat")*
    ls $DIR/AR7$(printf "%03d"  "$rat")* | wc -l
    ls $DIR/AR7$(printf "%03d" "$rat")* | sort | tail -n 1
    echo ---
done
