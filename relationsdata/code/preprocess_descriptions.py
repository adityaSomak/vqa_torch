import json

with open('region_descriptions.json') as json_file:
    data= json.load(json_file);
opFile = open('descriptionInText.txt','w');
for img in data: 
    for phr in img['regions']:
    	try:
        	opFile.write(phr['phrase']);
        	opFile.write('\n');
        except UnicodeEncodeError:
        	opFile.write(phr['phrase'].encode('ascii', 'ignore').decode('ascii'));
        	opFile.write('\n');
        	pass;

opFile.close();