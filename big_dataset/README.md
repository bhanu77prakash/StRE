## Big Dataset Experiments

<i> The experiment files are same as the category level experiments. Please find the inline comments from those files.</i>
#### Organization of the folder
```
1. dict_tokenizer.py - Generates required dictionary files for further processing by the Deep Models. 
2. bilstm_char.py - To run the character + attention model described in the paper.
3. bilstm_word.py - To run the word + attention model described in the paper.
4. word_char_attention.py - To run the character + word + attention model described in the paper.
5. retrain_wc_att.py - To run the retrained model (transfer learning) described in the paper.
6. test.py - To simply test the model on a input file.
7. data_sorter.py - a helper function to find the 20% data files as described for the big dataset experiments in the paper
8. data_mover.py - a helper function to move the 20% data files to a new folder to be trained on.
```
#### Requirements
1. nltk
2. sklearn
3. numpy==1.12.0
4. tqdm
5. sklearn_crfsuite
6. imblearn
7. pickle
8. keras==2.1.2
9. tensorflow==1.1.0

<i>Please see the codes for detailed inline comments.</i>
#### Dict Tokenizer

Input to the file should be processed file (using preprocess code in preprocess repository) which is generated from raw files. 
The preprocessed file is further processed by this file to generate embedding dictionaries needed by the deep models.
Saves 5 dictionaries in the given folder.

To run 

```
python dict_tokenizer.py <file_name>
```
#### Bilstm Char

Run the bilstm character model described in the paper.
It is implemented along with the attention module on the top of it.   
Reports the weighted F1 score, AUCROC and AUPRC value in order. 

To run 

```
usage: bilstm_char.py [-h] [--file FILE] [--epochs EPOCHS] [--model MODEL]
                      [--save_folder SAVE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           file containing input.txt
  --epochs EPOCHS       number of epochs
  --model MODEL         trained model
  --save_folder SAVE_FOLDER
                        directory to save model and test values

```

#### Bilstm Word

Run the bilstm word model described in the paper.
It is implemented along with the attention module on the top of it.   
Reports the weighted F1 score, AUCROC and AUPRC value in order. 

To run 

```
usage: bilstm_word.py [-h] [--file FILE] [--epochs EPOCHS] [--model MODEL]
                      [--save_folder SAVE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           file containing input.txt
  --epochs EPOCHS       number of epochs
  --model MODEL         trained model
  --save_folder SAVE_FOLDER
                        directory to save model and test values

```

#### Word Char Attention

Run the bilstm character + word model described in the paper.
It is implemented along with the attention module on the top of it.   
Reports the weighted F1 score, AUCROC and AUPRC value in order. 

To run 

```
usage: word_char_attention.py [-h] [--file FILE] [--epochs EPOCHS]
                              [--model MODEL] [--save_folder SAVE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           file containing input.txt
  --epochs EPOCHS       number of epochs
  --model MODEL         trained model
  --save_folder SAVE_FOLDER
                        directory to save model and test values

```

#### Retrain Wc Att

To run the transfer learning models described in the paper.
Input should be a model and file that needs to be retrained on the model.   
Reports the weighted F1 score, AUCROC and AUPRC value in order. 

To run 
```
usage: retrain_wc_att.py [-h] [--file FILE] [--epochs EPOCHS] [--model MODEL]
                         [--folder FOLDER]

optional arguments:
  -h, --help       show this help message and exit
  --file FILE      file containing input.txt
  --epochs EPOCHS  number of epochs
  --model MODEL    trained model
  --folder FOLDER  folder containing dictionaries

```

#### Test

Tests a given model on the given file. 
Input should be a model and file that needs to be tested on the model.   
Reports the weighted F1 score, AUCROC and AUPRC value in order. 

To run 
```
usage: test.py [-h] [--file FILE] [--model MODEL]
               [--folder FOLDER]

optional arguments:
  -h, --help       show this help message and exit
  --file FILE      file containing input.txt
  --model MODEL    trained model
  --folder FOLDER  folder containing dictionaries

```

#### Running the codes

1. Run the dict_tokenizer on the file
2. Run the required model on the file given the correct location of the dictionaries generated.
3. To run on the files in a directory, ./run.sh <folder_name>(for word_char_att), ./run_char.sh <folder_name>(for bisltm_char), ./run_word.sh <folder_name>(for bilstm_word)