import urllib.request
import os

os.makedirs('raw_data/train')
os.makedirs('raw_data/test')

# Download train data
f = open('data/train.txt', 'r')
f1 = f.readlines()
for url in f1:
    sign = url.split('/')[-2]
    filename = url.rstrip('\n').split('/')[-1]
    print(filename)
    if not os.path.exists(f'raw_data/train/{sign}'):
        os.mkdir(f'raw_data/train/{sign}')
    print(url)
    urllib.request.urlretrieve(url.rstrip('\n'), f'raw_data/train/{sign}/{filename}')

# Download test/validation data
f = open('data/test.txt', 'r')
f1 = f.readlines()
for url in f1:
    sign = url.split('/')[-2]
    filename = url.rstrip('\n').split('/')[-1]
    print(filename)
    if not os.path.exists(f'raw_data/test/{sign}'):
        os.mkdir(f'raw_data/test/{sign}')
    print(url)
    urllib.request.urlretrieve(url.rstrip('\n'), f'raw_data/test/{sign}/{filename}')
