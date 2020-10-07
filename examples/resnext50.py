import taso as ts
import onnx
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--result_file', type=str, default='resnext_time.txt', metavar='S',
        help='File to store times')
    parser.add_argument('--alpha', type=float, default=1.0,
        help='Threshold value')
    parser.add_argument('--iter', type=int, default=100,
        help='Number of iterations for backtracking search')
    return parser.parse_args()

def resnext_block(graph, input, strides, out_channels, groups):
    w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    w2 = graph.new_weight(dims=(out_channels,t.dim(1)//groups,3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    w3 = graph.new_weight(dims=(2*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*2):
        w4 = graph.new_weight(dims=(out_channels*2,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return graph.relu(graph.add(input, t))

graph = ts.new_graph()
input = graph.new_input(dims=(1,3,224,224))
weight = graph.new_weight(dims=(64,3,7,7))
t = graph.conv2d(input=input, weight=weight, strides=(2,2),
                 padding="SAME", activation="RELU")
t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="SAME")
for i in range(3):
    t = resnext_block(graph, t, (1,1), 128, 32)
strides = (2,2)
for i in range(4):
    t = resnext_block(graph, t, strides, 256, 32)
    strides = (1,1)
strides = (2,2)
for i in range(6):
    t = resnext_block(graph, t, strides, 512, 32)
    strides = (1,1)
strides = (2,2)
for i in range(3):
    t = resnext_block(graph, t, strides, 1024, 32)
    strides = (1,1)

#old_onnx = ts.export_onnx(graph)
#onnx.save(old_onnx, "resnext50.onnx")

old_time = graph.run_time()

args = get_args()
new_graph = ts.optimize(graph, alpha=args.alpha, budget=args.iter)

new_time = new_graph.run_time()
print("Run time of original graph is: {}".format(old_time))
print("Run time of optimized graph is: {}".format(new_time))

with open(args.result_file, "a") as f:
    f.write("{}\t{}\n".format(old_time, new_time))
