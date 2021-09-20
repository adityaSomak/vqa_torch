import os


def getAllPossibleAnswers(answerfile):
    answers = []
    with open(answerfile,'r') as ansFile:
        for line in ansFile:
            tokens = line.split("\t")
            if int(tokens[1]) > 10:
                answers.append(tokens[0])
    return answers[-1000:]

pslDataRootDir = "/home/somak/VQA_densecap/"
topAnswers = set(getAllPossibleAnswers(pslDataRootDir+"train/answers.txt"))
print topAnswers
dataDirectory = pslDataRootDir+"/pslData/test"
for qidFile in os.listdir(dataDirectory):
    qidFile = dataDirectory + "/" + qidFile
    if os.path.isdir(qidFile):
        continue
    newQIdFile = open(qidFile + ".txt", 'w')
    with open(qidFile, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if line == "" or line.startswith("#"):
                newQIdFile.write(line+"\n")
                continue
            tokens = line.split("\t")
            if tokens[0] == "word":
                if tokens[1] not in topAnswers:
                    continue
            newQIdFile.write(line + "\n")
    newQIdFile.close()
    os.remove(qidFile)
