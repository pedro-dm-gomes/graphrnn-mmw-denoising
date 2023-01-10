import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from datasets.labelled_mmW_split import MMW as Dataset_mmW
from datasets.labelled_mmW_split_eval import MMW as Dataset_mmW_eval
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


parser = argparse.ArgumentParser()
""" --  Training  hyperparameters --- """
parser.add_argument('--gpu', default='0', help='Select GPU to run code [default: 0]')
parser.add_argument('--data-dir', default='/home/uceepdg/profile.V6/Desktop/Datasets/mmW/interpolated_dataset', help='Dataset directory')
parser.add_argument('--batch-size', type=int, default=8, help='Batch Size during training [default: 16]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 200000]')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate [default: 1e-4]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients[default: 5.0 or 1e10 no clip].')
parser.add_argument('--restore-training', type= int , default = 0 , help='restore-training [default: 0=False 1= True]')

""" --  Save  hyperparameters --- """
parser.add_argument('--save-cpk', type=int, default=500, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--save-summary', type=int, default=30, help='Iterations to update summary [default: 20]')
parser.add_argument('--save-iters', type=int, default=1000, help='Iterations to save examples [default: 100000]')

""" --  Model  hyperparameters --- """
parser.add_argument('--model', type=str, default='GraphRNN_cls', help='Simple model or advanced model [default: advanced]')
parser.add_argument('--graph_module', type=str, default='Simple_GraphRNNCell', help='Simple model or advanced model [default: Simple_GraphRNNCell]')
parser.add_argument('--out_channels', type=int, default=32, help='Dimension of feat [default: 64]')
parser.add_argument('--num-samples', type=int, default=4, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=30, help='Length of sequence [default: 30]')
parser.add_argument('--num-points', type=int, default=1000, help='Number of points [default: 1000]')
parser.add_argument('--step-length', type=float, default=0.1, help='Step length [default: 0.1]')
parser.add_argument('--log-dir', default='outputs', help='Log dir [default: outputs/mmw]')
parser.add_argument('--version', default='v0', help='Model version')
parser.add_argument('--down-points1', type= int , default = 2 , help='[default:2 #points layer 1')
parser.add_argument('--down-points2', type= int , default = 2*2 , help='[default:2 #points layer 2')
parser.add_argument('--down-points3', type= int , default = 2*2*2, help='[default:2 #points layer 3')
parser.add_argument('--context-frames', type= int , default = 1, help='[default:1 #contex framres')

""" --  Additional  hyperparameters --- """
parser.add_argument('--bn_flag', type=int, default=1, help='Do batch normalization[ 1- Yes, 0-No]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]') 
parser.add_argument('--drop_rate', type=float, default=0.5, help='Dropout rate in second last layer[default: 0.0]') 

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

BN_FLAG = True if args.bn_flag == 1 else False
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
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
#write input arguments organize
LOG_FOUT.write("\n ========  Training Log ========  \n")
LOG_FOUT.write(":Input Arguments\n")
for var in args.__dict__:
	LOG_FOUT.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )


"""  Load Dataset """
#Load training dataset
train_dataset = Dataset_mmW(root=args.data_dir,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train=True)
#Load Test Dataset
test_dataset = Dataset_mmW_eval(root=args.data_dir,
                        seq_length= args.seq_length,
                        num_points=args.num_points,
                        train= False)

def get_classification_metrics(predicted_labels, ground_truth_labels, batch_size, seq_length,num_points,context_frames ): 
  """
  Gets predicted labels and gdt labels (numpy) and returns accuracy, F1 and Recall in numpy
  In : predicted_labels : (batch , seq_length, num_points, 2)
       ground_truth_labels : (batch , seq_length, num_points, 1) 
  Out: accuracy, f1_score, recall (1)
  """
  
  # cut context frames 
  predicted_labels = predicted_labels[:,context_frames:,:,:]
  ground_truth_labels = ground_truth_labels[:,context_frames:,:,:]
  
  predicted_labels = np.argmax(predicted_labels, axis=3)
  predicted_labels = np.expand_dims(predicted_labels, axis=3)
  correct = np.equal(predicted_labels,ground_truth_labels)
  accuracy = np.sum(correct.astype(int)) /( int(batch_size)  *  int(seq_length) *int(num_points) )

  true_positives = np.sum((predicted_labels == 1) & (ground_truth_labels == 1))
  false_positives = np.sum((predicted_labels == 1) & (ground_truth_labels == 0))
  true_negatives = np.sum((predicted_labels == 0) & (ground_truth_labels == 0))
  false_negatives = np.sum((predicted_labels == 0) & (ground_truth_labels == 1))

  return (accuracy, true_positives, false_positives, true_negatives, false_negatives)

def get_acurracy_tensor(predicted_labels, ground_truth_labels, batch_size, seq_length,num_points, context_frames ): 
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
    """ Visualize weights in terminal """
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


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay
    
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate,BASE_LEARNING_RATE) # CLIP THE LEARNING RATE!
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
    bn_decay = get_bn_decay(batch) 
    tf.summary.scalar('bn_decay', bn_decay)
        
    model_params = {'context_frames': int(int(args.context_frames)),
  	   	    'num_samples': int(args.num_samples),
  	   	    'graph_module': args.graph_module,
  	   	    'BN_FLAG':BN_FLAG, # decides if there is batch normalization
  	   	    'bn_decay':bn_decay,
  	   	    'out_channels':args.out_channels,
  	   	    'drop_rate': args.drop_rate,
  	   	    'sampled_points_down1':args.num_points/(args.down_points1),
  	   	    'sampled_points_down2':args.num_points/(args.down_points2),
  	   	    'sampled_points_down3':args.num_points/(args.down_points3)}

    pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, model_params)
    

    loss = MODEL.get_loss(pred, labels_pl, context_frames = model_params['context_frames'])
    tf.summary.scalar('loss', loss)

    # Calculate accuracy tensor
    accuracy = get_acurracy_tensor(pred, labels_pl,BATCH_SIZE,SEQ_LENGTH, NUM_POINTS, context_frames = model_params['context_frames'])
    tf.summary.scalar('accuracy', accuracy)

    # Get training operator
    learning_rate = get_learning_rate(batch) 
    #learning_rate = BASE_LEARNING_RATE
    tf.summary.scalar('learning_rate', learning_rate)
  
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, args.max_gradient_norm)
    train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=batch)    
    if is_training_pl == True :
	    # Clip gradient
	    params = tf.trainable_variables()
    if is_training_pl == False :
	    #Do not update parameters
	    params = tf.trainable_variables()
	    
    
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
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    # Init variables
    #init = tf.initialize_all_variables()
    if args.restore_training == False:
    	init = tf.global_variables_initializer()
    	sess.run(init, {is_training_pl: True})
    	ckpt_number = 0 
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
  	   'train_op': train_op,
  	   'params': params,
  	   'merged': merged,
  	   'step': batch}

    
    for epoch in range(ckpt_number, args.num_iters):
    	sys.stdout.flush()
    	
    	# Train one batch
    	train_one_batch(sess, ops,train_writer, epoch)
    	
    	# Save Checkpoint
    	if (epoch % args.save_cpk == 0):
    	  save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step = epoch)
    	  print("Model saved in file: %s" % save_path)
    	  #log_string("Model saved in file: %s" % save_path)
    	  
    	if (epoch % 1000 == 0):
    	    	   	  	  
    	  print(" **  Evalutate Test Data ** ")
    	  eval_one_epoch(sess, ops, test_writer, epoch)

    	  
    	  """ Restore After evaluation """
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
    	  
    	  # BUG !
    	  """
    	   In the "GraphRNN_OG" the weights are updated during evaluation and reseted after evaluation
    	   This bug should not happen in "GraphRNN_util" (double check)
    	   """
    	  
    	# Reload Dataset
    	if (epoch % 500 == 0 and epoch != 0):
    	  train_dataset = Dataset_mmW(root=args.data_dir,
                        		   seq_length=args.seq_length,
                        		   num_points=args.num_points,
                        		   train=True)
    	  print("[Dataset Reload] ",train_dataset )    	   	    	  
    	  


def train_one_batch(sess,ops,train_writer, epoch):
    """ Train one batch of trainin data """
    is_training = True
    
    # Load Batch Data    
    batch_data = get_batch(dataset=train_dataset, batch_size=args.batch_size)
    #batch = batch_data
    batch = np.array(batch_data)
    input_point_clouds = batch[:,:,:,0:3]
    input_labels = batch[:,:,:,3:4]
    

    feed_dict = {ops['pointclouds_pl']: input_point_clouds, ops['labels_pl']: input_labels, ops['is_training_pl']: is_training}
    
    pred, summary, step, train_op, loss, accuracy =  sess.run([ops['pred'], ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['acc']], feed_dict=feed_dict)
    
    print("[ %s  %03d ] Loss: %f\t  Accuracy: %f\t"%( args.version, epoch, loss, accuracy) )
    
    
    if (epoch % args.save_summary == 0 ):
    	train_writer.add_summary(summary, step)
    

                 
def eval_one_epoch(sess,ops,test_writer, epoch):
    """ Eval all sequences of test dataset """
    is_training = False
    
    nr_tests = len(test_dataset) 
    num_batches = nr_tests // BATCH_SIZE

    total_accuracy =0
    total_loss = 0
    Tp =0 #true positives total
    Tn =0
    Fp =0
    Fn =0

    # Load Test data
    for sequence_nr in range(0,nr_tests):
      test_seq = test_dataset[sequence_nr]    	
      test_seq =np.array(test_seq)
      input_point_clouds = test_seq[:,:,0:3]
      input_labels = test_seq[:,:,3:4]

      #Batch size problem - work around (this shitty way to do it)
      # TO DO:  FIX THIS!!!
      # We repeat the same sequence times the number of batches
      input_point_clouds = np.stack((input_point_clouds,) * args.batch_size, axis=0)
      input_labels = np.stack((input_labels,) * BATCH_SIZE, axis=0)

      feed_dict = {ops['pointclouds_pl']: input_point_clouds, ops['labels_pl']: input_labels, ops['is_training_pl']: is_training}

      pred, summary, step, train_op, loss, accuracy, params =  sess.run([ops['pred'], ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['acc'], ops['params'] ], feed_dict=feed_dict) 


      print("Loss  Accuracy: \t %f\t  %f\t"%( loss, accuracy) )
      test_writer.add_summary(summary, step)  

      #Visualize weights
      #print_weights(sess, params, 16) # FC2 Bias For GraphRNN_OG
      #print_weights(sess, params, 33) # FC2 Bias For GraphRNN_tf_util

      total_accuracy = total_accuracy + accuracy
      total_loss = total_loss + loss
      accuracy, true_positives, false_positives, true_negatives,false_negatives = get_classification_metrics(pred, input_labels, args.batch_size, args.seq_length,args.num_points, args.context_frames )
      Tp = Tp + (true_positives/num_batches) #it is the same sequence repeated for batch)
      Fp = Fp + (false_positives/num_batches)
      Tn = Tn + (true_negatives/num_batches)
      Fn = Fn + (false_negatives/num_batches)
      
    mean_loss = total_loss/ nr_tests
    mean_accuracy = total_accuracy/ nr_tests
    precision = Tp / ( Tp+Fp)
    recall = Tp/(Tp+Fn)
    f1_score =2 * ( (precision * recall)/(precision+recall) )
    
     
    print('**** EVAL: %03d ****' % (epoch))
    print("[Mean] Loss  Accuracy: %f\t  %f\t"%( mean_loss, mean_accuracy) )
    print("\nPrecision: ", precision, "\nRecall: ", recall, "\nF1 Score:", f1_score)
    print(' -- ')
    
        
    # Write to File
    #log_string('****  %03d ****' % (epoch))
    log_string('%03d  eval mean loss, accuracy: %f \t  %f \t' % (epoch, mean_loss , mean_accuracy))
    if not np.isnan(precision) and not np.isnan(recall) and not np.isnan(f1_score):
    	log_string('Precision %f Recall, F1 Score: %f \t  %f \t' % (precision, recall , f1_score))

    
                  
if __name__ == "__main__":
    train()
    LOG_FOUT.close()        




