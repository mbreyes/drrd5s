#!/bin/bash

#directory="/path/to/directory"   # Replace with your directory path
if [[ $# -eq 0 ]]; then
    echo "Please provide the directory path as an argument."
    echo "Usage: ./check_file_count.sh /path/to/directory"
    exit 1
fi

directory=$1 
expected_count=$(ls -1 "${directory}"/*.00? 2>/dev/null | wc -l)
last_extension=$(ls -1 "${directory}"/*.00? 2>/dev/null | tail -n 1 | grep -oE '[0-9]{3}$')
actual_count=$(ls -1 "${directory}"/*.00${last_extension} 2>/dev/null | wc -l)

if [[ "${expected_count}" -eq "${last_extension}" && "${expected_count}" -eq "${actual_count}" ]]; then
    echo "Number of files is correct."
else
    echo "Number of files is incorrect."

    # Find the missing file
    for ((i=1; i<=${last_extension}; i++)); do
        padded_number=$(printf "%03d" $i)
        if [[ ! -f "${directory}/file.${padded_number}" ]]; then
            echo "Missing file: file.${padded_number}"
            break
        fi
    done
fi
