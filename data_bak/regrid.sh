#!/bin/bash

for infile in ./zg/*.sel; do
  [ ! -f "$infile" ] && continue
  
  echo "正在处理: $infile"
  
  outfile="${infile%.sel}_re.nc"
  
  cdo remapbil,../target.txt "$infile" "$outfile"
  
  if [ $? -eq 0 ]; then
    echo "  -> 输出至: $outfile"
  else
    echo "  -> 处理失败: $infile" >&2
  fi
done

echo "批量处理完成！"