import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.estimator import regression 

img_size = 28
LR = 1e-3
MODEL_NAME = 'notMNIST-{}-{}.model'.format(LR, '2conv-basic')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('model loaded!')

train = np.load_data('training_data.npy')
test = np.load_data('testing_data.npy')


X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size,1 )
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, img_size, img_size,1 )
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


convnet  = input_data(shape=[None, img_size, img_size, 1], name=input)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax') 
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy')

model = tflearn.DNN(convnet, tensorboard_dir='log')

model.fit({'input': X}, {'targets': Y}, n_epoch=3,
	validation_set=({'input': test_x}, {'targets': test_y}), 
	snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save('train_model.model')