from gensim import models

def cleanup_w2v_based():
	allPredicates = set()
	with open('allPredicates.txt') as predicate_file:
	    for line in predicate_file:
			line = line.strip().lower()
			allPredicates.add(line);

	w2v_model = models.word2vec.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin',binary=True);
	print "Model Loaded..."
	with open('finalSortedPredicates.txt','w') as predicate_file: 
		for pred in allPredicates:
			tokens = pred.split(" ");
			invalid = False
			for token in tokens:
				if token not in w2v_model.vocab:
					invalid = True
			if not invalid:
				predicate_file.write(pred);
				predicate_file.write("\n");



removedRelations = set()		
with open('../input/Relations_Annotation - AllRelations.tsv','r') as predicate_file:
	for line in predicate_file:
		line  = line.split("\t")
		if len(line) == 2 and line[1].strip() == "0":
			removedRelations.add(line[0].strip())

finalRelations = []
with open('../input/Relations_Annotation - AllRelations.tsv','r') as predicate_file:
	for line in predicate_file:
		line  = line.split("\t")
		rel = line[0].strip()
		relWs = rel.split(" ")
		if len(relWs[-1]) == 1 or relWs[-1] == "th" or " th " in relWs:
			continue
		if line[1].strip() != "0" and rel != "" :
			finalRelations.append(rel)

with open('../input/filteredAnnotatedPredicates.txt','w') as new_predicate_file: 
	for pred in finalRelations:
		new_predicate_file.write(pred+"\n");