total="$(ls $1 | wc -l)"
counter=0
for filename in "$1/"*;do
counter=$((counter+1))
echo "Processing file $counter/$total"
python new_retrain_wc_att.py --file "$filename" --model "$2" --epochs 15 --folder "$3" >> "$4"
done
