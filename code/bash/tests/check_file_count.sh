#!/bin/bash

prefix=AR6

if [[ $# -eq 0 ]]; then
    echo "Please provide the directory path as an argument."
    echo "Usage: ./check_file_count.sh /path/to/directory"
    exit 1
fi

#fix the part below:
#it has to include the prefix and fix the way it parses the path (exclude the \

directory="$1"   # Directory path passed as argument
expected_count=$(ls -1 "${directory}"/*.00? 2>/dev/null | wc -l)
last_extension=$(ls -1 "${directory}"/*.00? 2>/dev/null | tail -n 1 | grep -oE '[0-9]{3}$')
actual_count=$(ls -1 "${directory}"/*.00${last_extension} 2>/dev/null | wc -l)

last_extension="${last_extension#"${last_extension%%[!0]*}"}"  # Remove leading zeros

echo "Lastextension = $last_extension"
echo "Exp count = ${expected_count}"
echo "Actual count = ${actual_count}"

if [[ "${expected_count}" -eq "${last_extension}" && "${expected_count}" -eq "${actual_count}" ]]; then
    echo "Number of files is correct."
else
    echo "Number of files is incorrect."

    # Find the missing file
    for ((i=1; i<=last_extension; i++)); do
        padded_number=$(printf "%03d" $i)
        if [[ ! -f "${directory}/file.${padded_number}" ]]; then
            echo "Missing file: file.${padded_number}"
            break
        fi
    done
fi
