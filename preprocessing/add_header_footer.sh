#!/bin/bash
FILE="$1"
folder="$1"

if [[ -d "$1" ]]; then
    for filename in "$1/"*; do
        echo "$filename"
        choice="$(head -n 1 $filename)"
        if [ "$choice" = '<?xml version="1.0" ?><page>' ]
        then
	       tail -n +2 "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"
            sed -i '1s/^/<page>\n/' "$filename"
        fi
	    cat "header" "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"
        cat  "$filename" "footer" > "$filename.tmp" && mv "$filename.tmp" "$filename"
    done
elif [[ -f "$1" ]]; then
	filename="$1"
    echo "$filename"
    choice="$(head -n 1 $filename)"
    if [ "$choice" = '<?xml version="1.0" ?><page>' ]
    then
	   tail -n +2 "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"
        sed -i '1s/^/<page>\n/' "$filename"
    fi
	cat "header" "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"
    cat  "$filename" "footer" > "$filename.tmp" && mv "$filename.tmp" "$filename"
else
    echo "$1 is not valid"
    exit 1
fi

#for filename in "$1/"*; do
#	echo "$filename"
#	if [ $2 == 'remove' ]
#	then 
#	tail -n +2 "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"
#	sed -i '1s/^/<page>\n/' "$filename"
#	fi
#	cat "header" "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"
#	cat  "$filename" "footer" > "$filename.tmp" && mv "$filename.tmp" "$filename"
#done
