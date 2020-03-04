import numpy as np
import os
import glob
import keras

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    
    def __init__(self, data_path, batch_size=16, seqLength=3, featureLength=1024, shuffle=True):
        """Initialization"""
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.seqLength = seqLength
        self.featureLength = featureLength
        self.shuffle = shuffle

        self.signs = sorted([name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))])
        self.n_classes = len(self.signs)

        filenames = []
        for sign in self.signs:
            pathname = os.path.join(data_path, sign, '*.npy')
            featurefiles = sorted(glob.glob(pathname))
            for featurefile in featurefiles:
                filenames.append(featurefile)
        self.samples = np.vstack(filenames)
        print(self.samples.shape)
        print("Number of signs = {0}".format(len(self.signs)))
        print("Number of samples = {0}".format(len(self.samples)))
        # self.samples = pd.DataFrame(sorted(glob.glob(data_path + "/*/*.npy")), columns=["path"])

        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch"""
        
        return int(np.floor(len(self.samples) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data"""
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        
        self.indexes = np.arange(len(self.samples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    def randind(self, N, n):
        s = [i*N//n for i in range(n)]
        s.append(N)
        nums = []
        for i in range(len(s)-1):
            ri = np.random.randint(s[i], s[i+1])
            nums.append(ri)

        return nums


    def __data_generation(self, indexes):
        """Generates data containing batch_size samples' # X : (batch_size, seqLength, featureLength)"""
        
        # Initialization
        dim = (self.seqLength, self.featureLength)
        X = np.empty((self.batch_size, *dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            file = self.samples[index][0]
            sign = file.split("/")[-2]
            # print(file, sign)
            signAsIndex = self.signs.index(sign)
            s = np.load(file)
            # print(s.shape)
            ri = self.randind(len(s), self.seqLength)
            # print(ri)
            rows = s[ri]
            X[i,] = rows
            y[i] = signAsIndex

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



if __name__ == "__main__":
    data_path, batch_size, featureLength, seqLength, shuffle = 'cnn_features/train', 5, 1024, 5, True

    datagen = DataGenerator(data_path,
                            batch_size=batch_size,
                            featureLength=featureLength,
                            seqLength=seqLength,
                            shuffle=shuffle)
    X, y = datagen[0]
    print(X)
    print(y)