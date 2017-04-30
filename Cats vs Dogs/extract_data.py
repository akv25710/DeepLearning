import cv2
import numpy as np 
from random import shuffle
import glob

img_size = 50
LR = 1e-3

animals = ["Cats","Dogs"]
training_data = []
testing_data = []

def extract_training_data(animal):

	files = glob.glob("train\\%s\\*"%animal)
	for file in files:
		print file
		img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size,img_size))
		if animals.index(animal) == 0:
			x = [1, 0]
		else:
			x = [0, 1]
		training_data.append([np.array(img), np.array(x)])


def extract_testing_data():

	files = glob.glob("test\\*")
	for file in files:
		#print file
		img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size,img_size))
		file = file.split(".")[-2]
		file = file[5:]
		print file
		testing_data.append([np.array(img), np.array(file)])


for animal in animals:
	extract_training_data(animal)
	

np.save('training_data.npy', training_data)
extract_testing_data()
np.save('testing_data.npy', testing_data)


