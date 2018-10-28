import jieba
from sys import argv


def is_a_num(value):
	try:
		float(value)
	except ValueError:
		return False
	else:
		return True


def cut_word(word):
	if is_a_num(word):
		return []
	else:
		#cut = jieba.cut(word, cut_al=True)
		#return list(set(cut))
		return list([x.encode('utf-8') for x in jieba.cut(word)])
		

if __name__ == "__main__":
	with open(argv[1],'r') as f1:
		txt = f1.readlines()

	word_bag = cut_word(''.join([x.strip() for x in txt]))
	
	word_count = {}
	for n in word_bag:
			if n.strip() == '':
					continue
			if n in word_count.keys():
					word_count[n] += 1
			else:
					word_count[n] = 1


	for k,v  in word_count.items():
			if v > 1:
				print k + '\t' + str(v)

