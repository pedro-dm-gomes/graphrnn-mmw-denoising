import os
import numpy as np


class MMW(object):
    def __init__(
        self,
        root="/home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset",
        seq_length=100,
        num_points=200,
        train=False,
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []

        log_nr = 0
        # print(" MMW DATASET ")
        root = root + '/' +str(num_points)
        if not(train):

            npy_files = [
                "labels_run_3.npy",
                "labels_run_4.npy",
                "labels_run_6.npy",
                "labels_run_7.npy",
                "labels_run_8.npy",
                "labels_run_9.npy",
                "labels_run_10.npy",
                "labels_run_11.npy",
                "labels_run_12.npy",
            ]
            
            # For each run calculate the limit
            for run in npy_files:
            	file_path = os.path.join(root, run)
            	npy_run = np.load(file_path)
            	#print("[LOAD EVAL] 30% File_path", file_path)
            	npy_run = npy_run[0]
            	
            	# Cut 70% of frames
            	d_len = int(npy_run.shape[0]*0.7)
            	npy_run = npy_run[d_len:]
            	
            	run_size = npy_run.shape[0]
            	start = 0
            	end = start + seq_length
            	while end < run_size:
            		npy_data = npy_run[start:end, :, :]
            		self.data.append(npy_data)
            		start = start + seq_length
            		end = start + seq_length
            
            print("Test  data", np.shape(self.data) )
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, nr):


        # select a seqeunce
        nr_seq = len(self.data)
        rand = nr
        log_data = self.data[rand]
        total_lenght = len(log_data)

        # print("nr frames:", total_lenght)
        start_limit = total_lenght - self.seq_length
        start = 0

        #print("[Seq] %d (of %d)  [%d - %d] (of %d)"% (rand, nr_seq, start, start + self.seq_length, total_lenght) )

        cloud_sequence = []
        cloud_sequence_color = []

        for i in range(start, start + self.seq_length):
            pc = log_data[i]

            npoints = pc.shape[0]
            # sample_idx = np.random.choice(npoints, self.num_points, replace=False)

            cloud_sequence.append(pc)

        points = np.stack(cloud_sequence, axis=0)

        return points
