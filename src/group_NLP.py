from sys import argv
from scipy.stats import fisher_exact
from random import randint

with open("md_tag.txt",'r') as f:
	txt = f.readlines()

group_tag = {}
for line in txt:
	l = line.strip().split()
	
	group_tag[l[0]] = l[1:]
	


with open(argv[1],'r') as f:
	txt = f.readlines()

with open('table_id.txt','r') as f:
	tid_list = [x.strip() for x in f.readlines()]

with open(argv[1],'r') as f:
	txt = f.readlines()

sid_list = [line.split('\t')[0] for line in txt]
sid_list = set(sid_list)

vec_dict = {}
seq_len = 180

target = {}


word_list = [str(x) for x in range(1, 5193)]
for n,line in enumerate(txt):
	l = line.strip().split('\t')
	sid, tid, vec, num = l
	vec = vec.split(',')

	if not group_tag.has_key(sid):
		continue

	if n % 1000 == 0:
		print n, sid

	tags = [int(x) for x in group_tag[sid]]

	for wid in word_list:
		if wid in vec:
			if target.has_key(wid):
				z1,z2,z3,z4 = zip(*target[wid])
				target[wid] = [[y1+1-x,y2+x, y3, y4] for x,y1,y2,y3,y4 in zip(tags, z1,z2,z3,z4)]
			else:
				target[wid] = [[1-x,x,0,0] for x in tags]
		elif randint(0,10) == 1:
			if target.has_key(wid):
				z1,z2,z3,z4 = zip(*target[wid])
				target[wid] = [[y1,y2, y3+1-x,y4+x] for x,y1,y2,y3,y4 in zip(tags, z1,z2,z3,z4)]
			else:
				target[wid] = [[0,0,1-x,x] for x in tags]


output = ""
print 'start processing target dict...'
for k, vs in target.items():
	for i,v in enumerate(vs): 
			if v[0] + v[1] <= 10:
				continue
			oddsratio, pvalue = fisher_exact([v[:2], v[2:]])
			output += '\t'.join([str(x) for x in [k, i, oddsratio, pvalue] + v]) + '\n'

print 'process target dict done!'

with open('output.csv','w') as f:
	f.writelines(output)
print 'write output.csv done!'
