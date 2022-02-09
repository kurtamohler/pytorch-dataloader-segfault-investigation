export CUDA_HOME=/usr/local/cuda-11.1.0

if [ ! -d $CUDA_HOME ]; then
  echo "Error: CUDA directory does not exist: $CUDA_HOME"
  echo "       Please install it"
  exit 1
fi

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LDFLAGS="$LDFLAGS -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath-link,$CUDA_HOME/lib64 -L$CUDA_HOME/lib64"
