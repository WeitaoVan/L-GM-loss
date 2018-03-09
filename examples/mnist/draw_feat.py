import numpy as np
import matplotlib
matplotlib.use('Agg')
#from matplotlib.pyplot import plot,savefig,text,show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys, os
os.environ['GLOG_minloglevel'] = '2'
caffe_root = '/home/zhongyy/caffe/'
sys.path.insert(0, caffe_root + 'python')
#from caffe.proto import caffe_pb2
import caffe

caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net('examples/mnist/extract_train_features.prototxt', 
    'examples/mnist/lenet_iter_%s.caffemodel' % sys.argv[2], caffe.TEST)
weights = net.params['ip2'][0].data
#print net.params.items()
print weights
#exit()
K = weights.shape[1]

import leveldb
# 'lr1_margin1_decay1'
db_feat = leveldb.LevelDB('examples/mnist/feat_train_' + sys.argv[1])
db_label = leveldb.LevelDB('examples/mnist/label_train')
p = [np.zeros((0,K))]*10

for k,v in db_feat.RangeIter():
    datum = caffe.proto.caffe_pb2.Datum.FromString(v)
    arr = caffe.io.datum_to_array(datum).squeeze()
    label = caffe.proto.caffe_pb2.Datum.FromString(db_label.Get(k))
    label = int(caffe.io.datum_to_array(label)[0][0][0])
    #plot(arr)
    p[label] = np.vstack((p[label], arr))
    #print arr, label
    #print np.dot(weights, arr.T)
    #print datum
    #break

colors = ['b','g','r','c','m','y','k','#7f00ff','#ff7f00','#00ff7f']
if K == 2:
    for i in range(10):
        #print p[i].shape
        plt.plot(p[i][:,0], p[i][:,1], '.', color = colors[i])
        plt.text(np.mean(p[i][:,0]), np.mean(p[i][:,1]), '%d' % i, color = 'w')
        plt.text(weights[i,0], weights[i,1], '%d' % i, color = '0.5')
elif K == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        #ax.scatter(p[i][:,0], p[i][:,1], p[i][:,2], '.', c=colors[i], s=5, depthshade=True, alpha=0.5, edgecolors='none')
        ax.plot(p[i][:,0], p[i][:,1], p[i][:,2], '.', c=colors[i], alpha=0.1)
        ax.text(np.mean(p[i][:,0]), np.mean(p[i][:,1]), np.mean(p[i][:,2]), '%d' % i, color = 'w')
        ax.text(weights[i,0], weights[i,1], weights[i,2], '%d' % i, color = '0.5')
    #plt.show()
#plt.axis('equal')

#caffe.reset_all()
plt.savefig('train_' + sys.argv[1] + '.png')