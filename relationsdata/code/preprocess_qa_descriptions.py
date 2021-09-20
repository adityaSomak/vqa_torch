import json

#idsToTextDict ={}
with open('question_answers.json') as json_file:
    dataqa= json.load(json_file);

opFile = open('questionsAnswersPlainText.txt','w');
for img in dataqa: 
	id = img['id'];
	for qa in img['qas']:
		try:
			opFile.write(str(id)+'\t'+qa['question']+"\t"+str(qa['answer'])+"\n");
		except:
			opFile.write(str(id)+'\t'+qa['question'].encode('ascii', 'ignore').decode('ascii')+"\t"+str(qa['answer'])+"\n");
opFile.close();


opFile = open('regionDescriptionsPlainText.txt','w');

with open('region_descriptions.json') as json_file:
    data= json.load(json_file);

for img in data:
	id = img['id'];
	phrases =[];
	for phr in img['regions']:
		bb = [phr['x'],phr['y'],phr['height'],phr['width']];
		try:
			opFile.write(str(id)+'\t'+str(bb)+"\t"+str(phr['phrase'])+"\n");
		except:
			opFile.write(str(id)+'\t'+str(bb)+"\t"+phr['phrase'].encode('ascii', 'ignore').decode('ascii')+"\n");
		#phrases.append((bb,phr['phrase']));
opFile.close();
