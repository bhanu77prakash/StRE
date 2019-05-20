# StRE
Implementation of the paper StRE: Self Attentive Edit Quality Prediction in Wikipedia


#### Organization of the folder

1. Preprocessing - preprocess the raw data files to required format as described in the paper.
2. Baselines - The ORES++ models that are described in the paper are implemented here. 
3. Category experiments - The character, word, both, attention modules are implemented. Transfer learning is also implemented.
4. Big dataset - The same experiments as in category but scaled up for the big dataset. The retraining module is modified to handle for the files in a folder. 

#### Baselines

1. The ORES++ baseline presented in the paper is given in the Baselines folder
2. Interrank and ORES baselines are taken from the repository [lca4/interrank](https://github.com/lca4/interank).

If you find this code useful in your research then please cite

#### To Run

Please look into the individual folders for running the codes.

If you find this code useful in your research then please cite

```
@inproceedings{soumya2019stre,
  title={StRE: Self Attentive Edit Quality Prediction in Wikipedia},
  author={Sarkar, Soumya and Guda, Bhanu Prakash Reddy and Sikdar, Sandipan and Mukherjee, Animesh},
  booktitle={ACL},
  year={2019}
}
```