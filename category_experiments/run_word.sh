for file in "$1/"*; do
	echo "$file"
	python dict_tokenizer.py "$file"
	python bilstm_word.py --file "$file" --epochs 10 --save_folder ./word_models/ 
done
