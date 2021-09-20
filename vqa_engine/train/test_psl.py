import argparse
import json
import os
import os.path

from pslplus.models import vqamodel

from ds import datasetprovider as dsp
from relationprediction import RelationPredictionWrapper
import time

def writeTestData(qWriter, combinedTriplets, semanticQRelations, nounPairsInQ, possibleAnswers):
    # has_q(.,.,.)
    qStr = "# "
    for nounPair in nounPairsInQ:
        qStr += ",".join(nounPair[:2]) + ";"
    qWriter.write(qStr + "\n")
    for q_relation in semanticQRelations:
        relStr = "has_q\t"+",".join([q_relation.word1,q_relation.relation,
                                  q_relation.word2])+"\t"+str(q_relation.confidence)
        qWriter.write(relStr+"\n")
    # word(.)
    for z_ans in possibleAnswers:
        relStr = "word\t"+ z_ans
        qWriter.write(relStr+"\n")
    # has_story(.,.,.)
    for storyTriplet in combinedTriplets:
        relStr = "has_story\t" + ",".join([storyTriplet.word1, storyTriplet.relation,
                                        storyTriplet.word2])
        qWriter.write(relStr + "\t" + str(storyTriplet.confidence) + "\n")
    #qWriter.close()

def ignoreQuestion(question):
    if question.startswith("are ") or question.startswith("how many") \
        or question.startswith("do ") or question.startswith("does ") \
            or question.startswith("has ") or question.startswith("is ") \
            or question.startswith("was ") or question.startswith("should ") \
            or question.startswith("how ") or question.startswith("can you") \
            or question.startswith("could ") or question.startswith("has ") \
            or question.startswith("none of the above "):
        return True
    return False

# TODO: Use Stanford Scene Graph Parser
def run_stage1(test_imagecaptions_dir, psl_test_data_dir, startFrom, answers, qaData, totalQs):
    count = 0
    image_count = 0
    bad_questions = []
    for json_filename in os.listdir(test_imagecaptions_dir):
        image_name = json_filename.replace(".jpg.json", "")
        image_count += 1
        if image_count < startFrom:
            continue
        with open(test_imagecaptions_dir + "/" + json_filename) as data_file:
            start = time.clock()
            data = json.load(data_file)
            imgName = data['img_name']
            captions = data['captions']
            scores = data['scores']
            boxes = data['boxes']
            print str(image_count)+":: ImageName: " + imgName

            completed = False
            ignoreImageCount = 0
            questionsByImgName = dsp.getQuestionsByImageName(imgName, qaData)

            for question in questionsByImgName:
                print question.question
                if ignoreQuestion(question.question.lower()):
                    ignoreImageCount += 1
                if os.path.exists(psl_test_data_dir + "/" + str(question.qid) + ".txt"):
                    completed = True
                    break
            if completed or ignoreImageCount == len(questionsByImgName):
                count += 3
                continue

            captionParses = RelationPredictionWrapper.getAllCaptionParses(captions, 20)
            maxCaptions = min(21,len(captionParses))
            # [spatialRelations, allObjects] = RelationPredictionWrapper.getSpatialRelations(
            #    captions[0:maxCaptions], captionParses, boxes[0:maxCaptions], scores[0:maxCaptions])
            # print "Spatial Relations done"
            semanticRelations = RelationPredictionWrapper.getRelationsFromCaptions(
                captions[0:maxCaptions], captionParses, boxes[0:maxCaptions], None) # allObjects[0:maxCaptions])
            # list = []
            # for rel in semanticRelations:
            #    list.append(rel.__str__())
            # print ",".join(list)
            combinedTriplets = semanticRelations
            # RelationPredictionWrapper.combineRelations(semanticRelations, spatialRelations)

            questionsByImgName = dsp.getQuestionsByImageName(imgName, qaData)
            for question in questionsByImgName:
                count += 1
                if ignoreQuestion(question.question.lower()):
                    continue
                qid = question.qid
                [possibleAnswers, relevance] = dsp.getAllPossibleAnswersAndRelevance(imgName, qaData)
                if possibleAnswers == None:
                    possibleAnswers = answers
                [semanticQRelations, nounPairsInQ] = RelationPredictionWrapper.getRelationsFromQuestion(question.question)
                qidFile = open(psl_test_data_dir + "/" + str(qid) + ".txt", 'w')
                if len(semanticQRelations) == 0:
                    bad_questions.append(qid)
                writeTestData(qidFile, combinedTriplets, semanticQRelations, nounPairsInQ, possibleAnswers)
                qidFile.close()
                print str(time.clock() - start)
        if count % 1000 == 0:
            print "####Questions processed:" + str(count) + "/" + str(totalQs)
    print bad_questions

def run_stage3(argsdict):
    dataDirectory = argsdict['datadir']
    qaData = argsdict['qaData']
    outputDir = dataDirectory + "/" + "psl"
    opFile = open(argsdict['parentDir']+"/answers_"+argsdict['split']+".txt",'w')
    for aidFile in os.listdir(outputDir):
        with open(outputDir+"/"+aidFile, 'r') as f:
            line = f.readline()
            line = line.replace("\n", "")
            tokens = line.split("\t")
            answer = tokens[0][tokens[0].index("(")+1:-1]
            qid = int(aidFile.replace(".txt",""))
            question = qaData.questionsDict[qid][0]
            op = str(qid)+"\t"+answer+"\t"+tokens[1]+"\t"+question
            print op
            opFile.write(op+"\n")
    opFile.close()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("qaTestDir",help="Testing Directory. Expects images/ as a sub-directory.")
    parser.add_argument("pslDataRootDir",help="PSL Data Root Directory. Expects trn/ as sub-directory.")
    parser.add_argument("answerFile",help="All answers file")
    parser.add_argument("startFrom",help="start from image-index")
    parser.add_argument("-stage",default=1, type=int, help="Stage 1/2/3")
    parser.add_argument("-split",default="test",help="test/dev")
    parser.add_argument("-vqaprior",default="",help="sub-directory under PSL Data Root Directory")
    parser.add_argument("-frequencyprior", default="",help="Frequent answers by type, json file, under PSL Data Root Dir")

    args = parser.parse_args()
    print args

    # 1. Get the Test questions, training images and Dense Captions
    # Questions in: <pslDataRootDir>/test or <pslDataRootDir>/valid
    # Dense Captions in <qaTestDir>/densecap
    print "Loading QA data..."
    [qaData, totalQs] = dsp.getTestQAData(args.__dict__['qaTestDir'], args.__dict__['split'])
    print "Questions loaded..."
    test_imagecaptions_dir = args.__dict__['qaTestDir']+"/densecap"

    answers = dsp.getAllPossibleAnswers(args.__dict__['answerFile'])
    # 3. PSL Input test data folder
    psl_test_data_dir = args.__dict__['pslDataRootDir'] + "/"+ args.__dict__['split']
    if args.__dict__['split'] == 'test':
        psl_test_data_dir = args.__dict__['pslDataRootDir']+"/test"

    vqa_psl_rules = args.__dict__['pslDataRootDir']+"vqa_rules_g.txt"

    startFrom = int(args.__dict__['startFrom'])
    if args.__dict__['stage'] == 1:
        # 4. Iterate over dense-captions and questions --> O/p: Triplets.
        run_stage1(test_imagecaptions_dir, psl_test_data_dir, startFrom, answers, qaData, totalQs)
    elif args.__dict__['stage'] == 2:
        # 5. Iterate over Triplets per question --> Use PSL to answer.
        answersByType = dsp.getAnswersByType(args.__dict__['pslDataRootDir']+"/top1000Answers.tsv")

        vqaPriorDir = None
        answerPriorByTypeDict = None
        if args.__dict__['vqaprior'] != "":
            vqaPriorDir = args.__dict__['pslDataRootDir'] + "/" + args.__dict__['vqaprior']
        elif args.__dict__['frequencyprior'] != "":
            freqPriorJSon = args.__dict__['frequencyprior']
            answerPriorByTypeDict = dsp.getAnswerPriorsByQType(freqPriorJSon)
        options = "" + vqa_psl_rules+" -datadir "+ psl_test_data_dir+ " -option infer"
        argsdict = {'pslcode':vqa_psl_rules, 'datadir':psl_test_data_dir,
                    'vqaprior': vqaPriorDir, 'qaData': qaData,
                    'answerPriorByTypeDict': answerPriorByTypeDict,
                    'answersByType': answersByType,
                    'option':"infer", 'startFrom': startFrom }
        #print argsdict
        vqamodel.run(argsdict)
    else:
        argsdict = {'pslcode': vqa_psl_rules, 'datadir': psl_test_data_dir,
                    'parentDir': args.__dict__['pslDataRootDir'],
                    'qaData': qaData, 'split': args.__dict__['split'], 'startFrom': startFrom}
        run_stage3(argsdict)