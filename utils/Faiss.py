import faiss
import h5py
import torch
import json
import numpy as np

class Cluster():
    def __init__(self):
        self.ncentroids = 30
        self.niter = 10000
        self.verbose = True
        self.num_gpu = 1

    # input : 2D array  1D array
    def Cluster(self, x, vid):
        d = x.shape[1]
        # kmeans = faiss.Kmeans(d, self.ncentroids, niter=self.niter, verbose=self.verbose, gpu=self.num_gpu)
        kmeans = faiss.Kmeans(d, self.ncentroids, niter=self.niter, verbose=self.verbose)
        kmeans.train(x)
        D, I = kmeans.index.search(x, 1)  # distance assign

        # save as json
        dict = {}
        i = 0
        for id in vid:
            dict[id] = [int(I[i][0]), int(I[i + len(vid)][0])]
            i += 1
        json_str = json.dumps(dict)

        with open('/data/whs/object/Flickr/class.json', 'w') as json_file:
            json_file.write(json_str)

    # read the json file
    def getLabel(self):
        with open('/data/whs/object/Flickr/class.json', 'r') as json_file:
            dict = json.load(json_file)

            print(dict.get(''))





if __name__ == '__main__':
    cluster = Cluster()



