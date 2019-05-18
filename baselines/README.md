## ORES++

### Organization of the folder
```
1. data_preprocess.py - Converts the processed data to features and their ground truth labels. 
2. Classifiers.py - Runs various ML classifiers on the generated features and labels(by default runs only RF classifier, see code to run other classifiers). 
```
#### Requirements
1. nltk
2. sklearn
3. numpy
4. tqdm
5. sklearn_crfsuite
6. imblearn
7. aspell (For installing please see [https://github.com/WojciechMula/aspell-python])
8. empath (pip install empath)

#### Data Preprocess

Input to the file should be processed file (using preprocess code in preprocess repository) which is generated from raw files. 
The preprocessed file is further processed by this file to convert to features and labels. 
Saves features.npy and labels.npy in the current directory

To run 

```
python data_preprocess.py <file_name>
```
#### Classifiers

Contains various ML classifiers. 
Runs the classifiers on the generated features and labels. 

To run 

```
python Classifiers.py
```
<i>Please see the codes for detailed inline comments.</i>

#### Running the codes

1. Run the data_preprocess on the file
2. Run the classifiers on the file
3. To run on the files in a directory, ./run.sh <folder_name>