#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=~/caffe/examples/imagenet
DATA=~/caffe/data/ilsvrc12
TOOLS=~/caffe/build/tools

#TRAIN_DATA_ROOT=train/ #/path/to/imagenet/train/
#VAL_DATA_ROOT=val/ #/path/to/imagenet/val/
VAL_DATA_ROOT=test/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

#if [ ! -d "$TRAIN_DATA_ROOT" ]; then
#  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
#  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet training data is stored."
#  exit 1
#fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

#echo "Creating train lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    --encode_type png --encoded \
#    $TRAIN_DATA_ROOT \
#    $DATA/train.txt \
#    ilsvrc12_train_lmdb
#    #$EXAMPLE/ilsvrc12_train_lmdb

#echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    ilsvrc12_test_lmdb

echo "Done."
