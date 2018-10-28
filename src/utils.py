from sys import argv

f1 =  open(argv[1],'r') 


id_count = {}
word_count = {}
while True:
#for i in range(100):
	line = f1.readline()

	if line == "":
		break

	l = line.strip().split('\t')

	table_id = l[1]
	count = l[-1]

	if id_count.has_key(table_id):
		id_count[table_id] += 1
		word_count[table_id] += int(count)
	else:
		id_count[table_id] = 1
		word_count[table_id] = int(count)


for k,v in sorted(word_count.items(), key=lambda x:x[1]):
	print '\t'.join([str(x) for x in [k, v, id_count[k], float(v)/id_count[k]]]) 
