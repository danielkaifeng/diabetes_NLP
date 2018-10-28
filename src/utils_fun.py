import numpy as np

def init_dict(num_classes): 
		label_acc = {}
		label_total = {}
		for i in range(num_classes):
				label_acc[i] = 0
				label_total[i] = 0

		return label_acc, label_total


def print_acc_log(label_acc, label_total, symp_idx_dict, tp_tn):
	sysout_dict = {}
	l_acc_list = []
	for k,v in label_total.items():
		if v == 0:
				continue
		l_acc = round(label_acc[k]/float(v),2)
		l_acc_list.append(l_acc)

		if tp_tn == 1 and l_acc < 0.05:
				continue
		elif tp_tn == 2 and l_acc > 0.95:
				continue

		sysout =  '\t'.join([str(x) for x in [symp_idx_dict[k],k,v, label_acc[k], l_acc]])

		sysout_dict[sysout] = l_acc
	
	for k,v in sorted(sysout_dict.items(), key = lambda x:x[1], reverse=True):
			print k
	print '** acc mean: %f' % np.mean(l_acc_list)	

def convert_label_to_multi_hot(labels, num_classes):
		output_label = []
		for label_idx in labels:
				multi_hot = [0] * num_classes
				for i in label_idx:
						multi_hot[i] = 1
				output_label.append(multi_hot)

		return np.array(output_label)



def convert_one_label_to_multi_hot(labels, num_classes):
		multi_hot = [0] * num_classes

		for i in labels:
			multi_hot[i] = 1

		return multi_hot



