"""
load_data functions

support most common format of image, including jpa, png, bml, etc.

support RGB input and GRAY input

support split data into batches
"""
# Last Modified : 2020/10/17 jzy_ustc

import cv2
import os
import torch
from torchvision import transforms


def load_data(path: str, batch_size: int, train_image_sub_path="/train/", test_image_sub_path="/test/",
			  train_label_sub_path="/train_label.dat", test_label_sub_path="/test_label.dat",
			  split_data_and_label=":", split_char=',', grayscale=False):
	"""
	split_data_and_label :
		char to split image path and label
		example : for data "1.png:3", we should use ":" 
		# split_data_and_label = ":"

	split_char :
		split char for each data
		# split_char = ','
	"""
	train_image_path = path + train_image_sub_path
	train_label_path = path + train_label_sub_path
	test_image_path = path + test_image_sub_path
	test_label_path = path + test_label_sub_path

	# judge whether the path exists
	for path_judge in [path, train_image_path, test_image_path, train_label_path, test_label_path]:
		if not os.path.exists(path_judge):
			raise Exception('Error : Path \'' + path_judge + '\' not exist!')

	# transform label data file into list
	train_label_list = load_label_list(train_label_path, split_char, split_data_and_label)
	test_label_list = load_label_list(test_label_path, split_char, split_data_and_label)

	# load train image data
	train_data = load_all_image_and_label(train_image_path, batch_size, train_label_list, grayscale)
	test_data = load_all_image_and_label(test_image_path, batch_size, test_label_list, grayscale)

	return train_data, test_data


def load_one_image_RGB(path: str):
	image = cv2.imread(path)

	# transform input (H,W,C),[0,255] into (C,H,W),[0,1]
	trans = transforms.ToTensor()
	image_tensor = trans(image)

	return image_tensor


def load_one_image_GRAY(path: str):
	image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)

	# transform input (H,W),[0,255] into (1,H,W),[0,1]
	trans = transforms.ToTensor()
	image_tensor = trans(image)

	return image_tensor


def load_label_list(path: str, split_char: str, split_data_and_label: str):
	with open(path, 'rb')as file:
		label_str = file.read().decode()
		label_list_temp = label_str.split(split_char)
		label_list = {}
		for item in label_list_temp:
			image_name, label = item.split(split_data_and_label, 1)
			label_list[image_name] = label
		file.close()
	return label_list


def load_all_image_and_label(image_path: str, batch_size: int, label_list: dict, grayscale=False):
	for root, dirs, files in os.walk(image_path):
		data = []
		batch_image = []
		batch_label = []
		num = 0
		batch_id = 0
		for file in files:

			# load image
			if grayscale:
				batch_image.append(load_one_image_GRAY(image_path + file))
			else:
				batch_image.append(load_one_image_RGB(image_path + file))
			# load label
			batch_label.append(int(label_list.get(file)))
			num += 1

			if num % batch_size == 0:
				# transform to tensor
				image_tensor = torch.stack(batch_image, 0)
				label_tensor = torch.tensor(batch_label)
				data.append([image_tensor, label_tensor])
				# initial next batch
				batch_image = []
				batch_label = []
				batch_id += 1

		return data
