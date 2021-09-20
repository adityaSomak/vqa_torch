## VQA PSL AAAI 2018, Aditya, Yang, Baral 

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