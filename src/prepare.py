import jieba

def load_word_idx(filepath):
	word_idx = {}

	with open(filepath,'r') as f1:
		txt = f1.readlines()

	i = 1
	for line in txt:
			l = line.strip().split('\t')
			word = l[0]
			word_idx[word] = i
			i += 1

	return word_idx


def query_word_in_line(line, word_idx):
	word_iter = jieba.cut(line)
	idx_line = []

	for word in word_iter:
			if word in word_idx:
				idx_line.append(word_idx[word])
			else:
				idx_line.append(0)

	return idx_line

def read_a_file(filepath, word_idx):
	with open(filepath,'r') as f1:
		txt = f1.readlines()

	idx_tensor = []
	label_tensor = []
	for line in txt:
		idx_tensor.append(query_word_in_line(line, word_idx))
		label_tensor = []

	return idx_tensor, label_tensor

word_idx = load_word_idx("word_bag.txt")
print len(word_idx)


with open("data_file.list",'r') as f1:
	filelist = f1.readlines()[:5]

idx_buf = []
for infile in filelist:
	infile = infile.strip()
	#outfile = "../output_submit01/" + infile.split('/')[-1].replace('txt','ann')


	idx_tensor,label_tensor = read_a_file(infile, word_idx)
	idx_buf += idx_tensor

idx_output = ""
for n in idx_buf:
		idx_output += ','.join([str(x) for x in n]) + '\n'

with open("word_idx.txt",'w') as f2:
	f2.writelines(idx_output)





