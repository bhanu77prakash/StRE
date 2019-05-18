touch "$2"
mkdir temp_merging_folder
for filename in "$1/"*; do
	echo "$filename"
	tail -n +2 "$filename" | head -n -1 > "$filename.tmp" 
	mv "$filename.tmp" "temp_merging_folder"
done

cat "temp_merging_folder/"* > "$2"
rm -r "temp_merging_folder"
