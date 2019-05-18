for file in "./data/person/"*; do
	echo "$file"
	python dict_tokenizer.py "$file"
	python bilstm_word.py --file "$file" --epochs 10 --save_folder ./word_models/ >> word_results
done

for file in "./data/person_new/"*; do
	echo "$file"
	python dict_tokenizer.py "$file"
	python bilstm_word.py --file "$file" --epochs 10 --save_folder ./word_models/ >> word_results
done