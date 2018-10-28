
import json

with open("ann_total.list",'r') as f1:
	txt = f1.readlines()


word_class = {}

for line in txt:
		l = line.strip().split('\t')
		word = l[2]
		entity = l[1].split()[0]

		if word not in word_class.keys():
				word_class[word] = entity

	
json.dump(word_class,open("NER.json",'w'), ensure_ascii=False)
