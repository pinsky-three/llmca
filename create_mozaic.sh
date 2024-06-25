#!/bin/bash

main_dir="./saves"

videos=($(find "$main_dir" -name "*.mp4"))

num_videos=${#videos[@]}

if [ $num_videos -lt 1 ]; then
  echo "At least 1 video is required to create the mosaic."
  exit 1
fi

video_size=480

grid_size=$(echo "sqrt($num_videos)" | bc)
if (( $(echo "$grid_size * $grid_size < $num_videos" | bc) )); then
  grid_size=$(($grid_size + 1))
fi

mosaic_width=$(($video_size * $grid_size))
mosaic_height=$(($video_size * $grid_size))

filter="nullsrc=size=${mosaic_width}x${mosaic_height} [base];"

for i in "${!videos[@]}"; do
  x=$(($i % $grid_size * $video_size))
  y=$(($i / $grid_size * $video_size))
  filter+="[$i:v] setpts=PTS-STARTPTS, scale=${video_size}x${video_size} [video$i];"
  if [ $i -eq 0 ]; then
    filter+="[base][video$i] overlay=shortest=1:x=$x:y=$y [tmp$i];"
  else
    filter+="[tmp$(($i - 1))][video$i] overlay=shortest=1:x=$x:y=$y [tmp$i];"
  fi
done
filter+="[tmp$(($num_videos - 1))]null"

inputs=""
for i in "${!videos[@]}"; do
  inputs+="-i ${videos[$i]} "
done

ffmpeg $inputs -filter_complex "$filter" -c:v libx264 output_mosaic.mp4

if [ $? -eq 0 ]; then
  echo "Video mosaic created successfully: output_mosaic.mp4"
else
  echo "Error creating video mosaic."
fi
