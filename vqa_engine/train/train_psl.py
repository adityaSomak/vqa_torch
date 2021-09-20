import argparse
import json

from pslplus.models import vqamodel

from ds import datasetprovider as dsp
from relationprediction import RelationPredictionWrapper


def writeTrainingData(qidFile, combinedTriplets, semanticQRelations):
    # TODO: implement
    raise NotImplementedError

'''
    Have to call lua modules from here.
    Might deserve a separate file.
'''
def runDenseCaption(train_image_dir):
    # TODO: implement
    raise NotImplementedError


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("qaTrainDir",help="Training Directory. Expects images/ as a sub-directory.")
    parser.add_argument("pslDataRootDir",help="PSL Data Root Directory. Expects trn/ as sub-directory.")

    args = parser.parse_args()

    # 1. Get the training questions and Answers
    # is in /windows/drive2/For PhD/KR Lab/DATASETS/QA/VQA/train
    # 2. Get training images
    # is in /windows/drive2/For PhD/KR Lab/DATASETS/QA/VQA/train/images

    qaData = dsp.getTrainingQAData(args['qaTrainDir'])
    train_image_dir = args['qaTrainDir']+"/images"

    # 3. PSL training data folder
    psl_train_data_dir = args['pslDataRootDir']+"/trn"
    vqa_psl_rules = args['pslDataRootDir']+"vqa_rules.txt"

    # 4. Use Dense Captioning to get captions from images.
    outputJson = runDenseCaption(train_image_dir)
    # outputJson contains the imagename, boxes, captions, probabilities.

    with open(outputJson) as data_file:
        data = json.load(data_file)

    for result in data['results']:
        imgName = data['img_name']
        captions = data['captions']
        scores = data['scores']
        boxes = data['boxes']

        semanticRelations = RelationPredictionWrapper.getRelationsFromCaptions(captions)
        spatialRelations = RelationPredictionWrapper.getSpatialRelations(captions, boxes, scores)
        combinedTriplets = RelationPredictionWrapper.combineRelations(semanticQRelations, spatialRelations)

        questionsByImgName = dsp.getQuestionsByImageName(imgName, qaData)
        for question in questionsByImgName:
            qid = question.id
            most_probable_answer = dsp.getMostProbableAnswerById(qid, qaData)
            semanticQRelations = RelationPredictionWrapper.getRelationsFromQuestion(question.Q)
            qidFile = open(psl_train_data_dir+"/"+qid+".txt",'w')
            writeTrainingData(qidFile, combinedTriplets,semanticQRelations)
            qidFile.close()
    options = "" + vqa_psl_rules+" -datadir "+ psl_train_data_dir+ " -option learn"
    vqamodel.run(options)