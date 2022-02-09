export CUDA_HOME=/usr/local/cuda-11.1.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LDFLAGS="$LDFLAGS -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath-link,$CUDA_HOME/lib64 -L$CUDA_HOME/lib64"
