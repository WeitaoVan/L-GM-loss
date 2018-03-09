#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>&1 | tee examples/mnist/logs/rbf_sigma_LeNet+.log

#$@
