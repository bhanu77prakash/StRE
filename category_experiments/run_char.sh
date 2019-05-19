for filename in "$1/"*;do
echo "Processing file $filename"
python dict_tokenizer.py "$filename"
python bilstm_char.py --file "$filename" --epochs 10 --save_folder char_models/ 
done

# echo "\n\nTechnology --------------" >> char_results.txt
# echo "\n\nTechnology --------------" >> word_results.txt
# for filename in "data/technology/"*;do
# echo "Processing file $filename"
# python dict_tokenizer.py "$filename"
# python bilstm_char.py --file "$filename" --epochs 10 --save_folder char_models/ >> char_results.txt
# python bilstm_word.py --file "$filename" --epochs 10 --save_folder word_models/ >> word_results.txt
# done
