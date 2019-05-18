total="$(ls $1 | wc -l)"
IFS=$'\n' read -d '' -r -a lines < "$5"
counter=0
for filename in "${lines[@]}";do
counter=$((counter+1))
echo "Processing file $counter/$total"
python new_retrain_wc_att.py --file "$1$filename.scores.txt" --model "$2" --epochs 10 --folder "$3" >> "$4"
done
