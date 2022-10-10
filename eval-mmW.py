import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from datasets.labelled_mmW_augumented import MMW as Dataset_mmW
#from datasets.labelled_mmW import MMW as Dataset_mmW
from datasets.labelled_mmW_eval import MMW as Dataset_mmW_eval
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


parser = argparse.ArgumentParser()
""" --  Training  hyperparameters --- """
parser.add_argument('--gpu', default='0', help='Select GPU to run code [default: 0]')
parser.add_argument('--data-dir', default='/home/uceepdg/profile.V6/Desktop/Datasets/mmW/interpolated_dataset', help='Dataset directory')
parser.add_argument('--batch-size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 200000]')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate [default: 1e-4]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients[default: 5.0 or 1e10 no clip].')
parser.add_argument('--manual_checkpoint', type= int , default = 0 , help='restore-training [default: 0=False 1= True]')
parser.add_argument('--ckpt-step', type=int, default=200000, help='Manual Checkpoint step [default: 200000]')

""" --  Save  hyperparameters --- """
parser.add_argument('--save-cpk', type=int, default=500, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--save-summary', type=int, default=30, help='Iterations to update summary [default: 20]')
parser.add_argument('--save-iters', type=int, default=1000, help='Iterations to save examples [default: 100000]')

""" --  Model  hyperparameters --- """
parser.add_argument('--model', type=str, default='GraphRNN_ShortTerm_models', help='Simple model or advanced model [default: advanced]')
parser.add_argument('--out_channels', type=int, default=64, help='Dimension of feat [default: 64]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=30, help='Length of sequence [default: 20]')
parser.add_argument('--num-points', type=int, default=1000, help='Number of points [default: 1000]')
parser.add_argument('--step-length', type=float, default=0.1, help='Step length [default: 0.1]')
parser.add_argument('--log-dir', default='outputs', help='Log dir [default: outputs/mmw]')
parser.add_argument('--version', default='v1', help='Model version')
parser.add_argument('--down-points1', type= int , default = 2 , help='[default:2 #points layer 1')
parser.add_argument('--down-points2', type= int , default = 2*2 , help='[default:2 #points layer 2')
parser.add_argument('--down-points3', type= int , default = 2*2*2, help='[default:2 #points layer 3')

""" --  Additional  hyperparameters --- """
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]') 
parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout rate in second last layer[default: 0.0]') 

print("\n ==== GRAPH-RNN for MMW DATASET FOR POINT CLASSIFICATION  ====== \n")

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

# Define the GPU to run the code
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

# Flags
BATCH_SIZE = args.batch_size
NUM_POINTS = args.num_points
SEQ_LENGTH = args.seq_length

BASE_LEARNING_RATE = args.learning_rate
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


"""  Setup Directorys """
MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', args.model+'.py')
print("MODEL_FILE", MODEL_FILE)
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_DIR = os.path.join(LOG_DIR, args.model + '_'+ args.version)
print("LOG_DIR", LOG_DIR)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_eval.txt'), 'a')
#write input arguments organize
#LOG_FOUT.write("\n ========  Training Log ========  \n")
#LOG_FOUT.write(":Input Arguments\n")
#for var in args.__dict__:
#	LOG_FOUT.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )


"""  Load Dataset """
#Load Test Dataset
test_dataset = Dataset_mmW_eval(root=args.data_dir,
                        seq_length= args.seq_length,
                        num_points=args.num_points,
                        train= False)

def get_acurracy_numpy(predicted_labels, ground_truth_labels, batch_size, seq_length,num_points ): 
  """
  Gets predicted labels and gdt labels (numpy) and returns accuracy
  In : predicted_labels : (batch , seq_length, num_points, 2)
       ground_truth_labels : (batch , seq_length, num_points, 1) 
  Out: accuracy (1)
  """
  predicted_labels = np.argmax(predicted_labels, axis=3)
  predicted_labels = np.expand_dims(predicted_labels, axis=3)
  correct = np.equal(predicted_labels,ground_truth_labels)
  accuracy = np.sum(correct.astype(int)) /( int(batch_size)  *  int(seq_length) *int(num_points) )

  return accuracy   

def get_acurracy_tensor(predicted_labels, ground_truth_labels, batch_size, seq_length,num_points, context_frames): 
  """
  Gets predicted labels and gdt labels (numpy) and returns accuracy
  In : predicted_labels : Tensor (batch , seq_length, num_points, 2)
       ground_truth_labels : Tensor(batch , seq_length, num_points, 1) 
  Out: accuracy (1)
  """
  predicted_labels = predicted_labels[:,context_frames:, :, :]
  ground_truth_labels = ground_truth_labels[:,context_frames:, :, :]
  predicted_labels = tf.argmax(predicted_labels, axis =3)
  predicted_labels = tf.expand_dims(predicted_labels, axis = 3)
  ground_truth_labels = tf.cast(ground_truth_labels, tf.int32)
  predicted_labels = tf.cast(predicted_labels, tf.int32)
  correct = tf.equal( predicted_labels ,ground_truth_labels ) 
  accuracy = tf.reduce_sum( (tf.cast(correct,tf.float32) ) )/( batch_size* (seq_length)  *num_points )    

  return accuracy

def print_weights(sess, params, layer_nr):
    """ Visualize weights """
    layers = np.array(params)
    layers = layers[layer_nr]
    W = layers#[layer_nr]
    print("W", W)
    print("Layer[",layer_nr, "]", W, "\n")    

def get_batch(dataset, batch_size):
    """ load from the dataset at random """
    batch_data = []
    for i in range(batch_size):
        sample = dataset[0]
        batch_data.append(sample)
    return np.stack(batch_data, axis=0)

# Not Used
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay
    
# I also using gradient cliping
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-3) # CLIP THE LEARNING RATE!
    return learning_rate   
    
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    



def train():
  with tf.Graph().as_default():
    
    is_training_pl = tf.placeholder(tf.bool, shape=())
    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, SEQ_LENGTH, NUM_POINTS)

  
    print("pointclouds_pl", pointclouds_pl)
    print("labels_pl", labels_pl)
    print("is_training_pl:", is_training_pl)

    
    batch = tf.Variable(0)
    #bn_decay = get_bn_decay(batch) # Not implemented
    bn_decay= 0 
    tf.summary.scalar('bn_decay', bn_decay)
        
    model_params = {'context_frames': int(1),
  	   	    'num_samples': int(8),
  	   	    'bn_decay':bn_decay,
  	   	    'out_channels':args.out_channels,
  	   	    'drop_rate': args.drop_rate,
  	   	    'sampled_points_down1':args.num_points/(args.down_points1),
  	   	    'sampled_points_down2':args.num_points/(args.down_points2),
  	   	    'sampled_points_down3':args.num_points/(args.down_points3)}

    pred, point_cloud_sequence = MODEL.get_model(pointclouds_pl, is_training_pl, model_params)
    

    loss = MODEL.get_loss(pred, labels_pl, context_frames = model_params['context_frames'])
    tf.summary.scalar('loss', loss)

    # Calculate accuracy tensor
    accuracy = get_acurracy_tensor(pred, labels_pl,BATCH_SIZE,SEQ_LENGTH, NUM_POINTS, context_frames = model_params['context_frames'])
    tf.summary.scalar('accuracy', accuracy)

    # Get training operator
    #learning_rate = get_learning_rate(batch) 
    learning_rate = BASE_LEARNING_RATE
    tf.summary.scalar('learning_rate', learning_rate)
  
    params = tf.trainable_variables()
    """ NO optimizer no Training """
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours = 5)
    
    
    print(" Trainable paramenters: ")
    for layer in params:
    	print(layer)
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session( config =config)
    
    # Add summary writers
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'Test_Eval'))

    # Init variables
    #init = tf.initialize_all_variables()
    if args.manual_checkpoint == 1:
    	checkpoint_path = os.path.join(LOG_DIR, 'ckpt-%d'%args.ckpt_step)
    	print("checkpoint_path", checkpoint_path)
    	exit()
  
    	ckpt_number = args.ckpt_step 
    else:
    	# Restore Session
    	checkpoint_path_automatic = tf.train.latest_checkpoint(LOG_DIR)
    	ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
    	print ("\n** Restore from checkpoint ***: ", checkpoint_path_automatic)
    	saver.restore(sess, checkpoint_path_automatic)
    	ckpt_number=ckpt_number[11:]
    	ckpt_number=int(ckpt_number)
    	# change random seed
    	np.random.seed(ckpt_number)
    	tf.set_random_seed(ckpt_number)

    ops = {'pointclouds_pl': pointclouds_pl,
  	   'labels_pl': labels_pl,
  	   'is_training_pl': is_training_pl,
  	   'pred': pred,
  	   'loss': loss,
  	   'acc':accuracy,
  	   'params': params,
  	   'merged': merged,
  	   'step': batch}

    mean_loss, mean_accuracy = eval_all_test_sequences(sess, ops, test_writer, ckpt_number)
    
def eval_all_test_sequences(sess, ops, test_writer, ckpt_number):
    """ Eval all sequences of test dataset """
    is_training = False
    
    nr_tests = len(test_dataset) 
    num_batches = nr_tests // BATCH_SIZE

    total_accuracy =0
    total_loss = 0
    # Load Test data
    for sequence_nr in range(0,nr_tests):
    	test_seq = test_dataset[sequence_nr]    	
    	test_seq =np.array(test_seq)
    	input_point_clouds = test_seq[:,:,0:3]
    	input_labels = test_seq[:,:,3:4]

    	#Batch size problem - work around (this is bad way to do it)
    	# TO DO:  FIX THIS!!!
    	input_point_clouds = np.stack((input_point_clouds,) * args.batch_size, axis=0)
    	input_labels = np.stack((input_labels,) * BATCH_SIZE, axis=0)
    	
    	feed_dict = {ops['pointclouds_pl']: input_point_clouds, ops['labels_pl']: input_labels, ops['is_training_pl']: is_training}
    	
    	pred, step, loss, accuracy, params =  sess.run([ops['pred'], ops['step'], ops['loss'], ops['acc'], ops['params'] ], feed_dict=feed_dict) 

    	print("Loss  Accuracy: \t %f\t  %f\t"%( loss, accuracy) )
    	#test_writer.add_summary(summary, step)  
    	
    	#Visualize weights
    	print_weights(sess, params, 16) # FC2 Bias For GraphRNN_OG
    	#print_weights(sess, params, 33) # FC2 Bias For GraphRNN_tf_util
    	
    	total_accuracy = total_accuracy + accuracy
    	total_loss = total_loss + loss
	
    mean_loss = total_loss/ nr_tests
    mean_accuracy = total_accuracy/ nr_tests
    
          
    print('**** EVAL: %03d ****' % (ckpt_number))
    print("[Mean] Loss  Accuracy: %f\t  %f\t"%( mean_loss, mean_accuracy) )
    print(' -- ')
    
    # Write to File
    #log_string('****  %03d ****' % (epoch))
    log_string('%03d  eval mean loss, accuracy: %f \t  %f \t' % (ckpt_number, mean_loss , mean_accuracy))
    
    return mean_loss, mean_accuracy

    



if __name__ == "__main__":
    train()
    LOG_FOUT.close()        




