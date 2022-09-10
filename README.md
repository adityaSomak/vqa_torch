## VQA PSL AAAI 2018, Aditya, Yang, Baral 

```Warning: This may not have complete instructions to run the code. Especially, as it depends on dense captioning results etc.```

This repo contains necessary helper codes to reproduce the results in [Explicit Reasoning over End-to-End Neural Architectures for Visual Question Answering](https://arxiv.org/abs/1803.08896) which was presented
  in AAAI 2018. 
- Data creation: `matlab_codes/train.m` (for relation prediction matrices), `relationsdata/code/datapreparation.py` (for training data preparation)
- Running, use `vqa_engine/main.py` (check README) one can also use the [PSLQA code-base](https://github.com/adityaSomak/PSLQA) for this.

-----------------------------------------------------------
Before running `vqa_engine/main.py` (according to `vqa_engine/README`), please follow [PSLQA code-base](https://github.com/adityaSomak/PSLQA) and install the PSL engine (the pre-requisites of installing the pythonic PSL engine is reproduced here for convenience)

## PSL Engine Pre-requisites:
   - Gurobi 6.5.0 or Gurobi 6.5.2 (please visit [Gurobi Website](http://www.gurobi.com/academia/for-universities) for license and download information). Please
    note that academic licenses (multiple) are free.
   - install gensim using `pip install gensim` to load word2vec models.
   - install nltk using `pip install nltk`.
   - Other packages to install: `numpy, enum, fuzzywuzzy, sqlite3`

   For ConceptNet and Word2vec, download `conceptnet-numberbatch-201609_en_word.txt` and `GoogleNews-vectors-negative300.bin` and change the paths
   in W2VPredicateSimilarity.py.



**Installing the PSL Engine**:

  Run the following commands:
   - `git clone https://github.com/adityaSomak/PSLQA.git`
   - `cd PSLQA`
   - `sudo python setup.py sdist`
   - `sudo pip install --upgrade dist/PSLplus-0.1.tar.gz`

## Pre-processing
1. We process the images through jcjohnson/densecap: Dense image captioning in Torch (github.com) to get the dense captions per image. Output should be of the form. ```<image_id>\t<dense-caption>\t<confidence-score>```
2. Run the [CoAttn code](jiasenlu/HieCoAttenVQA (github.com)) on each image separately to get the neural priors.
3. ```preprocess_qa_descriptions.py``` takes the raw outputs from densecap and CoAttn, and just writes out the important columns required (basically processes json and creates tsv files).
4. Once you have the outputs, you need to sort the output from above by Image IDs. You can simply use bash shell commands such as `sort` for this. 
5. You need to run the syntactic dependency parser using corenlp and then add a column that will add the list of all connected word-pairs for ```sortedqaDependencies.txt```
  - Is the man playing with a dog?          
  - <list_of_noun_pairs>: man-3,dog-7;is-1,man-3; ....
Follow L237-247 comments in `datapreparation.py`, so that formats match. Next is noun-pair selection and relation prediction using heuristics
6. datapreparation.py takes in sortedqaDependencies.txt (CoAttn processed output) sortedregionDescriptionsDependencies.txt (densecap processed output).
  - It will produce something that will look like [this file](https://github.com/adityaSomak/vqa_torch/blob/main/relationsdata/data/nounPairPhraseQuestionTraining.txt)
7. Then use [test_psl.py](https://github.com/adityaSomak/vqa_torch/blob/main/vqa_engine/train/test_psl.py). 
  - Run stage 1 with proper command line arguments to get the triplets. The triplet output should look like [11.txt](https://github.com/adityaSomak/PSLQA/blob/master/pslplus/data/vqa_demo/expt2_aaai18/11.txt)
