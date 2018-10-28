#coding=utf-8

import json
from sys import argv
import re

#word_dict = json.load(open("NER.json",'r'), encoding="utf-8")
word_dict = json.load(open("NER.json",'r'))
"""
i = 0
for k,v in word_dict.items():
		print k, v
		i += 1

		if i >20:
				break

word = word.decode('utf-8')
if word in word_dict.keys():
		print word_dict[word] 
"""

def query(doc):
	doc = doc.decode('utf-8')
	output = ""
	i = 1

	pos_recorder = []
	for k in word_dict.keys():
			#k = k.encode('utf-8')
			if len(k) <=1: 
					continue

			symbol = [')','(','[',']','+']
			mark = 0
			for sbl in symbol:
				if k.find(sbl) != -1:
						mark = 1
						break
			if mark == 1:
					continue

			pos_idx = [(x.start(), x.end()) for x in re.finditer(k, doc)]
			idx = []

			for p2 in pos_idx:
				inside = False
	
				for p1 in pos_recorder:
					if p2[0] >= p1[0] and p2[1] <= p1[1]:
						inside = True
						break
				if not inside:
					pos_recorder.append(p2)
					idx.append(str(p2[0]) + ' ' + str(p2[1]))

			for pos in idx:
					output += 'T%d' % i + '\t' + word_dict[k] + ' ' + pos + '\t' + k + '\n'
					i += 1
	return output

def get_a_file(filepath, output_prefix):
	print output_prefix
	with open(filepath,'r') as f1:
		txt  = f1.readlines()

	#output = query(''.join([x.strip() for x in txt]))
	output = query(''.join(txt))
	output = output.encode('utf-8')

	with open(output_prefix,'w') as f2:
		f2.writelines(output)


with open("testfile.list",'r') as f1:
	filelist = f1.readlines()

for infile in filelist:
	infile = infile.strip()
	outfile = "../output_submit03/" + infile.split('/')[-1].replace('txt','ann')
	get_a_file(infile, outfile)



