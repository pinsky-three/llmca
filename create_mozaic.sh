#!/bin/bash

# Simplified script to create a video mosaic

main_dir="./saves"
output_file="output_mosaic.mp4"
video_size=480

# Find all mp4 videos in the saves directory
videos=($(find "$main_dir" -name "*.mp4" | head -n 16)) # Limit to 4 videos for testing

num_videos=${#videos[@]}
if [ $num_videos -lt 1 ]; then
  echo "At least 1 video is required to create the mosaic."
  exit 1
fi

grid_size=$(echo "sqrt($num_videos)" | bc)
if (( $(echo "$grid_size * $grid_size < $num_videos" | bc) )); then
  grid_size=$(($grid_size + 1))
fi

mosaic_width=$(($video_size * $grid_size))
mosaic_height=$(($video_size * $grid_size))

# Calculate the maximum duration among the input videos
max_duration=0
for video in "${videos[@]}"; do
  duration=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$video")
  if [ -z "$duration" ]; then
    echo "Unable to determine the duration of the video: $video"
    exit 1
  fi
  if (( $(echo "$duration > $max_duration" | bc -l) )); then
    max_duration=$duration
  fi
done

if (( $(echo "$max_duration == 0" | bc -l) )); then
  echo "Unable to determine the maximum duration of the videos."
  exit 1
fi

filter="nullsrc=size=${mosaic_width}x${mosaic_height}:duration=${max_duration} [base];"
for i in "${!videos[@]}"; do
  x=$(($i % $grid_size * $video_size))
  y=$(($i / $grid_size * $video_size))
  filter+="[$i:v] setpts=PTS-STARTPTS, scale=${video_size}x${video_size}, tpad=stop_mode=clone:stop_duration=$(echo "$max_duration - $(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 ${videos[$i]})" | bc) [video$i];"
  if [ $i -eq 0 ]; then
    filter+="[base][video$i] overlay=shortest=0:x=$x:y=$y [tmp$i];"
  else
    filter+="[tmp$(($i - 1))][video$i] overlay=shortest=0:x=$x:y=$y [tmp$i];"
  fi
done
filter+="[tmp$(($num_videos - 1))]null"

inputs=""
for i in "${!videos[@]}"; do
  inputs+="-i ${videos[$i]} "
done

ffmpeg -y $inputs -filter_complex "$filter" -c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p $output_file

if [ $? -eq 0 ]; then
  echo "Video mosaic created successfully: $output_file"
else
  echo "Error creating video mosaic."
fi
