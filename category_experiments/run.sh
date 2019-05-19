for filename in "$1/"*;do
echo "Processing file $filename"
python dict_tokenizer.py "$filename"
python word_char_attention.py --file "$filename" --epochs 15 
done

for filename in "$1/"*;do
echo "Processing file $filename"
python dict_tokenizer.py "$filename"
python word_char_attention.py --file "$filename" --epochs 15 
done
