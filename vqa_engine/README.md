**To Run VQA-model from command-line**:

Directory Structure:
- qatestdir: Expects generated Json files from DenseCaptioning software for each image under the subdirectory "densecap/"
- pslDataRootDir: Stage 1 output will be stored here. Stage 2 output will be stored
   under "psl" subdirectory.
- COMPOUND_NOUN_VOCAB_DIR: Expects the file "stats_conceptNet_vocab/allEnglishPhrasesin_cn5.5.txt" in it. These are the
compound nouns from ConceptNet5.5
- SIZE_COLOR_DICT_DIR: Expects the file "allSizeAdjectives.txt", for all size adjectives and
adverbs.
- REP_NET_PARAMS_DIR: The processed relations from visual genome should be in this directory 
under the name "data/processedFAPredicates.txt"

Prerequisties:
   - Use `pip install` to install the following packages: `pycorenlp, pyparsing, gensim, enum, fuzzywuzzy, nltk, networkx`
   - Also might have to use `pip install enum34`

Use:
   - `python2.7 main.py <qatestdir> <pslDataRootDir> <answerFile> -stage 1/2/3 -split test/dev <startFrom>`
        - answerFile: All the top frequent answers in Training Set.
   - Currently run `-stage 1`, `-stage 2`, `-stage 3` consecutively.

Example:
   - `qatestdir`: /data/somak/DATASETS/VQA/VQA_densecap
   - `pslDataRootDir`: /data/somak/DATASETS/VQA/VQA_densecap/pslData
   - `split`: test
   - `answerFile`: /data/somak/DATASETS/VQA/VQA_densecap/pslData/top1000Answers.tsv