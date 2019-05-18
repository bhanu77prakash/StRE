#!/bin/bash
set -e

folder="$1" 
for filename in "$1/"*; do
	echo "Executing compute_quality for $filename"
	python3 ./scripts/compute_quality.py --processes=48 "$filename" > "$filename.qualities.txt"
	echo "Executing extractor for $filename"
	python3 ./extractor.py "$filename" "$filename.extracted.txt"
	echo "Executing cleaner for $filename"
	python3 ./java_code_cleaner.py "$filename.extracted.txt" "$filename.cleaned.txt"
	echo "Executing parser for $filename"
	python3 ./parser.py "$filename" "$filename.temp" "$filename.qualities.txt" "$filename.cleaned.txt" "$filename.scores.txt"
	echo "Removing temporary files for $filename"
	rm "$filename.extracted.txt" "$filename.cleaned.txt" "$filename.temp"
	echo "Moving files"
	mv "$filename" "$filename."* "$2"
	echo "Done processing $filename"
done
