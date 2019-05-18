echo "Concept --------------" >> results.txt
for filename in "data/concept/"*;do
echo "Processing file $filename"
python dict_tokenizer.py "$filename"
python word_char_attention.py --file "$filename" --epochs 15 >> results.txt
done

echo "\n\nTechnology --------------" >> results.txt
for filename in "data/technology/"*;do
echo "Processing file $filename"
python dict_tokenizer.py "$filename"
python word_char_attention.py --file "$filename" --epochs 15 >> results.txt
done
