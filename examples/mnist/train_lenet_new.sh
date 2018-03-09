#!/usr/bin/env sh
set -e
./build/tools/caffe train --solver=examples/mnist/lenet_solver_new.prototxt -gpu 1 $@
#./build/tools/caffe train --solver=examples/mnist/lenet_solver_new.prototxt -gpu 1 -snapshot examples/mnist/lenet_iter_10000.solverstate $@
