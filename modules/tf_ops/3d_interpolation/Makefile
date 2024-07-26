CUDA_HOME := /apps/cuda/cuda-12.2
CUDA_HOME := /usr/local/cuda
CUDNN_HOME := /apps/cuda/cudnn-7.6.5
CUDNN_HOME := /usr/lib/x86_64-linux-gnu

TF_INC = `python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB = `python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"`
TF_LIB1 = /usr/local/lib/python3.6/dist-packages/tensorflow_core
TF_LIB2 = /usr/local/cuda/lib64

$(info var is $(TF_LIB))

LIBFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -L$(CUDNN_HOME) -L$(TF_LIB2) -lcudart -L$(TF_LIB1) -ltensorflow_framework -lcublas

arch = sm_61

LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$(CUDNN_HOME)/lib64
INCLUDE_PATH=$(CUDA_HOME)/include

all: tf_interpolate_so.so

tf_interpolate_so.so: tf_interpolate.cpp
	g++ -std=c++11 -shared -fPIC -o tf_interpolate_so.so tf_interpolate.cpp -I$(TF_INC) -I$(INCLUDE_PATH) $(LIBFLAGS) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm *.so
