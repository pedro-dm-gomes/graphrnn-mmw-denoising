import os
import numpy as np


class MMW(object):
    def __init__(
        self,
        root="/home/pedro/Desktop/Datasets/NPYs",
        seq_length=12,
        num_points=4000,
        train=True,
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []

        # print("seq_length",seq_length)
        # print("num_points",num_points)

        log_nr = 0
        # print(" MMW DATASET ")

        print("root folder", root)

        if train:
            npy_files = [
                "labels_run_4.npy",
                "labels_run_6.npy",
                "labels_run_8.npy",
                "labels_run_9.npy",
                #"labels_run_10.npy",
                "labels_run_11.npy",
                "labels_run_12.npy"
            ]

            for run in npy_files:
                file_path = os.path.join(root, run)
                npy_run = np.load(file_path)
                #npy_run = npy_run[:, :, :, ]
                print(" LOAD File_path", file_path)
                # print("npy_run", npy_run.shape)
                npy_run = npy_run[0]
                print("run shape", npy_run.shape)
                self.data.append(npy_run)

        else:

            npy_files = [
                "labels_run_7.npy",
                "labels_run_10.npy",
                "labels_run_3.npy"
            ]

            
            # For each run calculate the limit
            for run in npy_files:
            	#print ("run", run)
            	file_path = os.path.join(root, run)
            	npy_run = np.load(file_path)
            	print(" LOAD File_path", file_path)
            	npy_run = npy_run[0]
            	#print("run shape", npy_run.shape)
            	
            	run_size = npy_run.shape[0]
            	start = 0
            	end = start + seq_length
            	while end < run_size:
            		#print("Add sequence")
            		#print("start", start)
            		#print("end", end)
            		
            		npy_data = npy_run[start:end, :, :]
            		#print("npy_data", npy_data.shape)
            		self.data.append(npy_data)
            		
            		
            		start = start + seq_length
            		end = start + seq_length
            
            print("self.data", np.shape(self.data) )
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, nr):

        # print("Get sequence")
        nr_seq = len(self.data)
        # print("nr sequences:", nr_seq)

        # select random sequnce
        rand = nr
        log_data = self.data[rand]
        total_lenght = len(log_data)

        # print("nr frames:", total_lenght)

        start_limit = total_lenght - self.seq_length
        start = 0

        print("[Seq] %d (of %d)  [%d - %d] (of %d)"% (rand, nr_seq, start, start + self.seq_length, total_lenght) )

        cloud_sequence = []
        cloud_sequence_color = []

        for i in range(start, start + self.seq_length):
            pc = log_data[i]

            npoints = pc.shape[0]
            # sample_idx = np.random.choice(npoints, self.num_points, replace=False)

            cloud_sequence.append(pc)

        points = np.stack(cloud_sequence, axis=0)

        return points
