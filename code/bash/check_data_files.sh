#!/bin/bash

# check if a file name was provided as a command-line argument
if [ $# -eq 0 ]; then
  echo "Usage: $0 prefix"
  exit 1
fi

# the argument parsed is the prefix of each files
prefix=$1

for file_name in $(find ../../ -iname "$prefix*" -type f); do

    # assign the file name from the command-line argument to a variable
    # echo $file_name

    # checking if the file has more than one Start Date
    n=$(grep -c 'Start Date' $file_name)

    # check if a file name was provided as a command-line argument
    if [ $n -gt 1 ]; then
	echo $file_name
	echo "File $file_name has more than one start dates"
    fi
done
