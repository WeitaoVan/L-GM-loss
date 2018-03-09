# usage: ./train.sh 0 simple
set -x
GPUs=$1 # e.g. 0 (for single GPU)
NET=$2 # e.g. simple

EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

solver=${NET}/solver.prototxt
LOG="${NET}/logs/${NET}_${EXTRA_ARGS_SLUG}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# specify your caffe path here
../../build/tools/caffe train -gpu ${GPUs} \
    -solver ${solver}
set +x
