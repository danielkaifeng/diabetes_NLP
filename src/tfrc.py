#coding=utf8
import os
import tensorflow as tf
from sys import argv
import random

import json

def write_tfrecords(data,data2, y, writer):
	example = tf.train.Example(features=tf.train.Features(feature={
		"xn": tf.train.Feature(float_list=tf.train.FloatList(value=data2)),
		"x": tf.train.Feature(int64_list=tf.train.Int64List(value=data)),
		'y': tf.train.Feature(float_list=tf.train.FloatList(value=y))
	}))

	writer.write(example.SerializeToString())

def create_record(x_path, xn_path, y_path):
	train_writer = tf.python_io.TFRecordWriter("./data/train_numeric.tfrecords")
	test_writer = tf.python_io.TFRecordWriter("./data/test_numeric.tfrecords")

	f1 = open(x_path, 'r')
	f2 = open(y_path, 'r')
	f3 = open(xn_path, 'r')

	i = 1

	#head = f1.readline()
	while True:
	#for n in range(100):
			xline = f1.readline()
			yline = f2.readline()
			xnline = f3.readline()
			if xline == "" and yline == "":
					break

			lx = xline.strip().split(',')
			lxn = xnline.strip().split(',')
			ly = yline.strip().split('\t')
	
			ID = lx[0]
			assert ID == ly[0] 
			assert ID == lxn[0]

			data = [int(x) for x in lx[1:]]
			data2 = [float(x) for x in lxn[1:]]
			y = [float(x) for x in ly[1:]]

			#binary_label = convert_label_to_multihot(anno_json[ID],299)
			#data += binary_label

			train = True
			if random.randint(0,9) == 1:
					train = False

			if i % 200 == 1:
				print str(i) 
			i += 1

			if train:
				write_tfrecords(data, data2, y, train_writer)
			else:
				write_tfrecords(data, data2, y, test_writer)

	f1.close()
	f2.close()
	f3.close()
	
	train_writer.close()
	test_writer.close()

def convert_label_to_multihot(labels, num_class):
	# label with 7 class example: [3, 5]  ==>  [0,0,0,1,0,1,0]
	multihot_label = [0] * num_class
	for i in labels:
		multihot_label[int(i)] = 1

	return multihot_label



def read_and_decode(filename):
	print 'read and decode data...'
	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
									   features={
											'xn': tf.FixedLenFeature([217], tf.float32),
										   'x': tf.FixedLenFeature([47880], tf.int64),
										   #'x': tf.FixedLenFeature([47880], tf.float32),
										   'y' : tf.FixedLenFeature([5], tf.float32),
									   })

	x = tf.cast(features['x'], tf.int32)
	y = tf.cast(features['y'], tf.float32)
	xn = tf.cast(features['xn'], tf.float32)
	#x = features['x']
	#y = features['y']

	return x, xn, y




if __name__ == '__main__':
	#train_x_path = "data/train_x0.csv"
	train_y_path = "data/train_y.csv"

	#train_x_path = "data/sort_train160.csv"
	#test_x_path = "data/test160.csv"
	train_xn_path = "data/train_xu.csv"
	train_x_path = argv[1]

	#json_path = "data/transfer.json"
	#anno_json = json.load(open(json_path, 'r'))
	if False:
		with open(train_x_path, 'r') as f1:
			head = f1.readline()
			txt = f1.readlines()

		out = ""
		for xline in txt:
			lx = xline.strip().split(',')
	
			ID = lx[0]
			data = lx[1:]
			binary_label = convert_label_to_multihot(anno_json[ID],299)
			data += [str(x) for x in binary_label]

			out += ID + ',' + ','.join(data) + '\n'
			
		with open("data/train_160299.csv",'w') as f2:
		#with open("data/test_160299.csv",'w') as f2:
			f2.writelines(out)


	create_record(train_x_path, train_xn_path, train_y_path)



















