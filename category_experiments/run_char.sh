echo "company --------------" >> company_char_results.txt
echo "company --------------" >> company_word_results.txt
for filename in "data/company/"*;do
echo "Processing file $filename"
python dict_tokenizer.py "$filename"
python bilstm_char.py --file "$filename" --epochs 10 --save_folder char_models/ >> company_char_results.txt
python bilstm_word.py --file "$filename" --epochs 10 --save_folder word_models/ >> company_word_results.txt
python word_char_attention.py --file "$filename" --epochs 15 --save_folder company_both_models/ >> company_results.txt
done

# echo "\n\nTechnology --------------" >> char_results.txt
# echo "\n\nTechnology --------------" >> word_results.txt
# for filename in "data/technology/"*;do
# echo "Processing file $filename"
# python dict_tokenizer.py "$filename"
# python bilstm_char.py --file "$filename" --epochs 10 --save_folder char_models/ >> char_results.txt
# python bilstm_word.py --file "$filename" --epochs 10 --save_folder word_models/ >> word_results.txt
# done
