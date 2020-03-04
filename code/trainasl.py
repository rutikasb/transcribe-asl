import os
from pathlib import Path
from random import shuffle
from shutil import copyfile
import argparse
import numpy as np
import glob
from keras.preprocessing import image
from keras.applications import MobileNet
from keras.models import load_model, Model
from keras.applications.mobilenet import preprocess_input
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_generator import DataGenerator

class TrainAsl(object):
	def __init__(self, dataFolder: str) -> None:
		self.dataFolder = dataFolder
		path = Path(self.dataFolder)
		print(path.parent)
		self.trainFolder = path.parent / 'train'
		self.testFolder = path.parent / 'test'
		if not os.path.exists(self.trainFolder):
			os.mkdir(self.trainFolder)
		if not os.path.exists(self.testFolder):
			os.mkdir(self.testFolder)

	def train_lstm_jittered(self, features_path, sequencelength, epochs, num_features, num_classess):
	    model = Sequential()
	    model.add(LSTM(64, return_sequences=False,input_shape=(sequencelength, num_features),dropout=0.5))
	    model.add(Dense(32, activation='relu'))
	    model.add(Dropout(0.5))
	    model.add(Dense(num_classess, activation='softmax'))
	    optimizer = Adam(lr=1e-4)
	    metrics = ['accuracy']
	    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

	    callbacks = [ModelCheckpoint('video_LSTM.h5', monitor='val_loss', save_best_only=True, verbose=1)]

	    data_path, batch_size, featureLength, shuffle = os.path.join(features_path, 'trainFeatures'), 32, num_features, True
	    train_gen = DataGenerator(data_path,
	                            batch_size=batch_size,
	                            featureLength=featureLength,
	                            seqLength=sequencelength,
	                            shuffle=shuffle)

	    data_path, batch_size, featureLength, shuffle = os.path.join(features_path, 'testFeatures'), 16, num_features, False
	    val_gen = DataGenerator(data_path,
	                            batch_size=batch_size,
	                            featureLength=featureLength,
	                            seqLength=sequencelength,
	                            shuffle=shuffle)
	    
	    model.fit_generator(generator=train_gen,
	            validation_data=val_gen,
	            epochs=int(epochs),
	            workers=1,
	            use_multiprocessing=False,
	            verbose=1,
	            callbacks=callbacks)
	    return model


	def extract_cnn_features(self, input_dir, output_dir):
	    model=MobileNet(weights='imagenet',include_top=True)
	    base_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)
	    _, height, width, channels = base_model.input_shape

	    print(base_model.summary())

	    signs = sorted([name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))])
	    for sign in signs:
	        print("\nSign = {0}\n-------------------".format(sign))
	        videos = sorted([name for name in os.listdir(os.path.join(input_dir, sign)) if os.path.isdir(os.path.join(input_dir, sign, name))])
	        # For each video we will save the cnn features computed on each image
	        for video in videos:
	            print("Video = {0}".format(video))
	            features_path = f'{output_dir}/{sign}/{video}.npy'
	            if not os.path.exists(f'{output_dir}/{sign}'):
	                os.makedirs(f'{output_dir}/{sign}')

	            pathname = os.path.join(input_dir, sign, video, '*.jpg')
	            imagenames = sorted(glob.glob(pathname))
	            featuresList = []
	            for imagename in imagenames:
	                img = image.load_img(imagename, target_size=(height, width))
	                img_data = image.img_to_array(img)
	                img_data = np.expand_dims(img_data, axis=0)
	                img_data = preprocess_input(img_data)
	                mobilenet_feature = base_model.predict(img_data)
	                # print("featureLength = {}".format((mobilenet_feature.shape)))
	                featuresList.append(mobilenet_feature)
	            stackedFeatures = np.vstack(featuresList)
	            np.save(features_path, stackedFeatures)
	            print("{0} images processed\n".format(len(featuresList)))


	def splitTrainTest(self, splitRatio=0.8):
		path = Path(self.dataFolder)
		folders = os.listdir(self.dataFolder)
		folders = [os.path.join(self.dataFolder, f) for f in folders]
		for folder in folders:
			print(folder)
			dst = os.path.join(self.trainFolder, os.path.basename(folder))
			if not os.path.exists(dst):
					os.mkdir(dst)
			dst = os.path.join(self.testFolder, os.path.basename(folder))
			if not os.path.exists(dst):
					os.mkdir(dst)

			files = [os.path.join(folder,file) for file in os.listdir(folder) if file.endswith(".avi") or file.endswith(".mp4")]
			shuffle(files)
			split_index = int(splitRatio*len(files))
			train_files = files[:split_index]
			test_files = files[split_index:]
			for trainf in train_files:
				dst = os.path.join(self.trainFolder, os.path.basename(folder), os.path.basename(trainf))
				copyfile(trainf, dst)
			for testf in test_files:
				dst = os.path.join(self.testFolder, os.path.basename(folder), os.path.basename(testf))
				copyfile(testf, dst)
			




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train CNN+LSTM model for short video snippets')
	parser.add_argument('--dataFolder', default='../data/master', required=False, help='path to master data folder')
	parser.add_argument('--processedDataFolder', default='../processed_data', required=False, help='path to processed data folder')
	args = vars(parser.parse_args())
	processedTrainFolder = f'{Path(args["processedDataFolder"])}/train'
	processedTestFolder = f'{Path(args["processedDataFolder"])}/test'
	trainFeaturesFolder = f'{Path(args["processedDataFolder"])}/trainFeatures'
	testFeaturesFolder = f'{Path(args["processedDataFolder"])}/testFeatures'
	
	print("Processed Train folder: {}".format(processedTrainFolder))
	print("Processed Test folder: {}".format(processedTestFolder))
	print("Train features folder: {}".format(trainFeaturesFolder))
	print("Test features folder: {}".format(testFeaturesFolder))
    
	trainAsl = TrainAsl(args["dataFolder"])
	# trainAsl.splitTrainTest()
	# trainAsl.extract_cnn_features(processedTrainFolder, trainFeaturesFolder)
	# trainAsl.extract_cnn_features(processedTestFolder, testFeaturesFolder)

	num_features = 1024
	num_classes = len(os.listdir(f'{args["processedDataFolder"]}/trainFeatures'))
	seqLength = 7
	lstm_epochs = 1000
	model = trainAsl.train_lstm_jittered(args['processedDataFolder'], seqLength, lstm_epochs, num_features, num_classes)
