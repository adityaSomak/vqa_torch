from util.Parameters import *

inputDir = SIZE_COLOR_DICT_DIR;#"/windows/drive2/For PhD/KR Lab/ASU_Vision/VQA/VQA_torch/relationsdata/data/"
allSizeAdjectives = open(inputDir+ "allSizeAdjectives.txt", "r").readlines()

allColorNames = set()
with open(inputDir + "allColorNamesEnglish.txt", "r") as clrNamesFile:
	for line in clrNamesFile:
		if "(" in line:
			line = line[:line.rindex("(")].lower().strip()
		else:
			line = line.lower().strip()
		allColorNames.add(line)

allLocationNames = set(["right", "left", "up", "down"])

def matchesSzColorLocation(word):
	if word in allSizeAdjectives:
		return "size"
	if word in allColorNames:
		return "color"
	if word in allLocationNames:
		return "location"
	return None

def matchesPosTagPattern(postaggedPhrase, minWordPos, maxWordPos):
	noConjuctionFromHere = False
	for i in range(minWordPos, maxWordPos):
		if postaggedPhrase[i-1][1].startswith("N"):
			noConjuctionFromHere = True
		if postaggedPhrase[i-1][1] == "CC" and noConjuctionFromHere:
			return "invalid"
		if postaggedPhrase[i-1][1].startswith("J") or \
		postaggedPhrase[i-1][1] == "CC":
			continue;
		else:
			return "False"
	return "True"


def simplifyRelation(predictedRelation, pairInPhrase, postaggedPhrase, w2v_model):
	if predictedRelation is None:
		return None
	if predictedRelation.strip() == "has on":
		return "wearing"
	nounWsPhrase = map(lambda x: x[:x.rindex("-")],pairInPhrase)
	predictedRelationWs = map(lambda x: x.strip(), predictedRelation.split(" "))
	matchingRel = matchesSzColorLocation(nounWsPhrase[1])
	wordPositionsInPhrase = map(lambda x: int(x[x.rindex("-")+1:]),pairInPhrase)
	if matchingRel is None:
		return predictedRelation
	if matchingRel == "size" or matchingRel =="color":
		minWordPos = min(wordPositionsInPhrase)
		maxWordPos = max(wordPositionsInPhrase)
		
		newWs = []
		for W in predictedRelationWs:
			if W == "of":
				newWs.append("Of")
				continue
			if W in w2v_model.vocab:
				newWs.append(W)
		pattern = matchesPosTagPattern(postaggedPhrase, minWordPos, maxWordPos)
		if pattern == "invalid":
			return None
		if w2v_model.n_similarity(newWs, ["is"]) > 0.8 or \
		w2v_model.n_similarity(newWs, ["are"]) > 0.8 or \
		pattern == "True":
			return matchingRel
	if matchingRel == "location":
		if wordPositionsInPhrase[0] < wordPositionsInPhrase[1]:
			return matchingRel
	return predictedRelation



