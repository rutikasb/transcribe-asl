
echo "In folder: $1"
echo "Out folder: $2"

for folder in $1/*; do
    echo "$(basename "$folder")"
    mkdir -p -- "$2/$(basename "$folder")"
    for file in $folder/*; do
    	echo "$(basename "$file")"
    	echo "$2/$(basename "$folder")/""$(basename "$file")"".mp4"
    	ffmpeg -i "$file" -vcodec libx264 "$2/$(basename "$folder")/""$(basename "$file")"".mp4"
    done
done
