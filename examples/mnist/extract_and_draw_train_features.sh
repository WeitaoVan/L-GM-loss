rm -r examples/mnist/feat_train_$1
./build/tools/extract_features examples/mnist/lenet_iter_$2.caffemodel examples/mnist/extract_train_features.prototxt ip1 examples/mnist/feat_train_$1 600 leveldb GPU 1
python examples/mnist/draw_feat.py $1 $2