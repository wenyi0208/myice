#!/bin/bash

for infile in ./zg/*.nc
do
  if [ ! -f "$infile" ]; then
    continue
  fi

  echo "正在处理: $infile"

  outfile="${infile%.nc}.sel"

  cdo sellevel,50000. "$infile" "$outfile"

  if [ $? -eq 0 ]; then
    echo "  -> 输出至: $outfile"
  else
    echo "  -> 处理失败: $infile" >&2
  fi
done
echo "批量处理完成！"