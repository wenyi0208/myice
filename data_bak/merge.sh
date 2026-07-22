#!/bin/bash

for file in ./zg/*_re.nc; do
    prefix=$(echo "$file" | sed -E 's/(.*)_[0-9]{6}-[0-9]{6}_re\.nc$/\1/')
    echo "$prefix"
done | sort -u | while read prefix; do
    files=$(ls -1 ${prefix}_[0-9]*-[0-9]*_re.nc 2>/dev/null)
    file_count=$(echo "$files" | wc -l)
    
    if [ "$file_count" -gt 1 ]; then
        echo "合并 $file_count 个文件: $prefix"
        sorted_files=$(echo "$files" | sort)
        output="${prefix}_185001-201412_re.nc"
        cdo mergetime $sorted_files "$output"
    fi
done
