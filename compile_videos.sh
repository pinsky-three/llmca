#!/bin/bash

main_dir="./saves"

for subdir in "$main_dir"/*; do
    for dir in "$subdir"/*; do
        folder_name=$(basename "$dir")
        
        # echo "Revisando la carpeta: $folder_name"
        
        # echo "Archivos en $dir:"
        # ls -lv "$dir"
        
        if compgen -G "$dir/*.png" > /dev/null; then
            filelist="$dir/ffmpeg_filelist.txt"

            rm -f "$filelist"
            for f in $(ls "$dir"/*.png | sort -V); do
                # get only the filename
                f=$(basename "$f")
                echo "file '$f'" >> "$filelist"
            done
            
            output_file="$subdir"_"$folder_name.mp4"
            ffmpeg -y -r 6 -f concat -safe 0 -i "$filelist" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "$output_file"
            
            if [ $? -eq 0 ]; then
                echo "Video creado para la carpeta: $folder_name"
            else
                echo "Error al crear el video para la carpeta: $folder_name"
            fi
        else
            echo "No se encontraron archivos PNG en la carpeta: $folder_name" > /dev/null
        fi
    done
done
