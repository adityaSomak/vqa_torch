from Parameters import *

def isCompoundNoun(phrase):
    return phrase in compoundNounsDict

compoundNounsDict = set()
vocabDir = COMPOUND_NOUN_VOCAB_DIR #"/windows/drive2/For PhD/KR Lab/ASU_Vision/VQA/VQA_torch/vqa_engine/"
with open(vocabDir+"stats_conceptNet_vocab/allEnglishPhrasesin_cn5.5.txt") as cnPhrases:
    for line in cnPhrases:
        tokens =line.strip().split("_")
        if len(tokens) == 2:
            compoundNounsDict.add(" ".join(tokens))