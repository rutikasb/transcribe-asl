import urllib.request
import os

def download(dtype):
    os.makedirs(f'raw_data/{dtype}')
    f = open(f'data/{dtype}.txt', 'r')
    f1 = f.readlines()
    for url in f1:
        sign = url.split('/')[-2]
        filename = url.rstrip('\n').split('/')[-1]
        print(filename)
        if not os.path.exists(f'raw_data/{dtype}/{sign}'):
            os.mkdir(f'raw_data/{dtype}/{sign}')
        print(url)
        urllib.request.urlretrieve(url.rstrip('\n'), f'raw_data/{dtype}/{sign}/{filename}')

# Download train & validation data
for dtype in ['train', 'test']:
    download(dtype)
