# Script to run for all files in a folder. 
# For individual please run the commands in order. 

for filename in "$1/"*;do
echo "Processing file $filename"
python3 data_preprocess.py "$filename" 
python3 Classifiers.py
done

