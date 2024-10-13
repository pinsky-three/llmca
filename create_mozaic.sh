#!/bin/bash

main_dir="./saves"
output_file="output_mosaic.mp4"
video_size=480
temp_dir="./temp"

mkdir -p "$temp_dir"

videos=($(find "$main_dir" -name "*.mp4"))

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

max_duration=0
for video in "${videos[@]}"; do
  duration=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$video")
  if [ -z "$duration" ]; then
    echo "Unable to determine the duration of the video: $video"
    exit 1
  fi
  duration_ms=$(echo "$duration * 1000" | bc | awk '{printf "%d\n", $0}')
  if (( $duration_ms > $max_duration )); then
    max_duration=$duration_ms
  fi
done

if (( $max_duration == 0 )); then
  echo "Unable to determine the maximum duration of the videos."
  exit 1
fi

intermediate_videos=()
for i in "${!videos[@]}"; do
  video_duration=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "${videos[$i]}")
  video_duration_ms=$(echo "$video_duration * 1000" | bc | awk '{printf "%d\n", $0}')
  stop_duration=$(( $max_duration - $video_duration_ms ))
  if (( $stop_duration > 0 )); then
    intermediate_video="$temp_dir/video_$i.mp4"
    ffmpeg -y -i "${videos[$i]}" -vf "tpad=stop_mode=clone:stop_duration=${stop_duration}ms" -c:v libx264 -preset veryslow -crf 18 "$intermediate_video"
    if [ -f "$intermediate_video" ]; then
      intermediate_videos+=("$intermediate_video")
    else
      echo "Failed to create intermediate video: $intermediate_video"
      exit 1
    fi
  else
    intermediate_videos+=("${videos[$i]}")
  fi
done

filter="nullsrc=size=${mosaic_width}x${mosaic_height}:duration=$(echo "scale=3; $max_duration / 1000" | bc) [base];"
for i in "${!intermediate_videos[@]}"; do
  x=$(($i % $grid_size * $video_size))
  y=$(($i / $grid_size * $video_size))
  filter+="[${i}:v] setpts=PTS-STARTPTS, scale=${video_size}x${video_size} [video$i];"
  if [ $i -eq 0 ]; then
    filter+="[base][video$i] overlay=shortest=1:x=$x:y=$y [tmp$i];"
  else
    filter+="[tmp$(($i - 1))][video$i] overlay=shortest=1:x=$x:y=$y [tmp$i];"
  fi
done
filter+="[tmp$(($num_videos - 1))]null"

inputs=""
for i in "${!intermediate_videos[@]}"; do
  inputs+="-i ${intermediate_videos[$i]} "
done

ffmpeg -y $inputs -filter_complex "$filter" -c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p $output_file

if [ $? -eq 0 ]; then
  echo "Video mosaic created successfully: $output_file"
else
  echo "Error creating video mosaic."
fi

rm -rf "$temp_dir"
