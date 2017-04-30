import cv2
import numpy as np 
import glob
from random import shuffle


characters = ["A","B","C","D","E","F","G","H","I","J"]


training_data = []
testing_data = []

def get_onehot(n):
	val=[]
	for i in xrange(10):
		if i==n:
			val.append(1)
		else:
			val.append(0)
	return val


def extract_trainingdata(character):
	files = glob.glob("notMNIST_large\\%s\\*"%character)
	
	for file in files:
		print file
		try:
			img = cv2.imread(file,cv2.IMREAD_GRAYSCALE) 
			val = get_onehot(characters.index(character))
			training_data.append([np.array(img), np.array(val)])
		except IOError as e:
			print ('Could not read:', file, ':', e)

		
	
	

def extract_testingdata(character):
	files = glob.glob("notMNIST_small\\%s\\*"%character)
	
	for file in files:
		print file
		try:
			img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
			val = get_onehot(characters.index(character))
			testing_data.append([np.array(img), np.array(val)])
		except IOError as e:
			print ('Could not read:', file, ':', e)



for character in characters:
	extract_testingdata(character)
	extract_trainingdata(character)

shuffle(training_data)
np.save('training_data.npy', training_data)
np.save('testing_data.npy', testing_data)
