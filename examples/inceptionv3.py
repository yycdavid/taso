import taso as ts
import onnx
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--result_file', type=str, default='bert_time.txt', metavar='S',
        help='File to store times')
    return parser.parse_args()


def inceptionA(graph, input, inputC, channels):
	weight1 = graph.new_weight(dims=(64, inputC, 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(48,inputC, 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(64,48, 5, 5))
	t2 = graph.conv2d(input=t2, weight=weight3, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(64,inputC, 1, 1))
	t3 = graph.conv2d(input=input, weight=weight4, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight5 = graph.new_weight(dims=(96,64, 3, 3))
	t3 = graph.conv2d(input=t3, weight=weight5, strides=(1,1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(96,96, 3, 3))
	t3 = graph.conv2d(input=t3, weight=weight6, strides=(1,1),
                 padding="SAME", activation="RELU")
	t4 = graph.avgpool2d(input=input, kernels=(3, 3), strides=(1,1), padding="SAME", activation = "RELU")
	weight7 = graph.new_weight(dims=(channels,inputC, 1, 1))
	t4 = graph.conv2d(input=t4, weight=weight7, strides=(1,1),
                 padding="SAME", activation="RELU")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	t12 = graph.concat(1, inputs1)
	inputs2 = list()
	inputs2.append(t3)
	inputs2.append(t4)
	t34 = graph.concat(1, inputs2)
	inputs3 = list()
	inputs3.append(t12)
	inputs3.append(t34)
	return graph.concat(1, inputs3)



def inceptionB(graph, input):
	weight1 = graph.new_weight(dims=(384, input.dim(1), 3, 3))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(2, 2),
                 padding="VALID", activation="RELU")
	weight2 = graph.new_weight(dims=(64, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(96, 64, 3, 3))
	t2 = graph.conv2d(input=t2, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(96, 96, 3, 3))
	t2 = graph.conv2d(input=t2, weight=weight4, strides=(2, 2),
                 padding="VALID", activation="RELU")
	t3 = graph.avgpool2d(input=input, kernels=(3, 3), strides=(2, 2), padding="VALID")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	t12 = graph.concat(1, inputs1)
	inputs2 = list()
	inputs2.append(t12)
	inputs2.append(t3)
	return graph.concat(1, inputs2)



def inceptionC(graph, input, channels):
	weight1 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(channels, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(channels, channels, 1, 7))
	t2 = graph.conv2d(input=t2, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(192, channels, 7, 1))
	t2 = graph.conv2d(input=t2, weight=weight4, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight5 = graph.new_weight(dims=(channels, input.dim(1), 1, 1))
	t3 = graph.conv2d(input=input, weight=weight5, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(channels, channels, 7, 1))
	t3 = graph.conv2d(input=t3, weight=weight6, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight7 = graph.new_weight(dims=(channels, channels, 1, 7))
	t3 = graph.conv2d(input=t3, weight=weight7, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight8 = graph.new_weight(dims=(channels, channels, 7, 1))
	t3 = graph.conv2d(input=t3, weight=weight8, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight9 = graph.new_weight(dims=(192, channels, 1, 7))
	t3 = graph.conv2d(input=t3, weight=weight9, strides=(1, 1),
                 padding="SAME", activation="RELU")
	t4 = graph.avgpool2d(input=input, kernels=(3, 3), strides=(1, 1), padding="SAME", activation="RELU")
	weight10 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t4 = graph.conv2d(input=t4, weight=weight10, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	t12 = graph.concat(1, inputs1)
	inputs2 = list()
	inputs2.append(t3)
	inputs2.append(t4)
	t34 = graph.concat(1, inputs2)
	inputs3 = list()
	inputs3.append(t12)
	inputs3.append(t34)
	return graph.concat(1, inputs3)




def inceptionD(graph, input):
	weight1 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(320, 192, 3, 3))
	t1 = graph.conv2d(input=t1, weight=weight2, strides=(2, 2),
                 padding="VALID", activation="RELU")
	weight3 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(192, 192, 1, 7))
	t2 = graph.conv2d(input=t2, weight=weight4, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight5 = graph.new_weight(dims=(192, 192, 7, 1))
	t2 = graph.conv2d(input=t2, weight=weight5, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(192, 192, 3, 3))
	t2 = graph.conv2d(input=t2, weight=weight6, strides=(2, 2),
                 padding="VALID", activation="RELU")
	t3 = graph.maxpool2d(input=input, kernels=(3,3), strides=(2,2), padding="VALID")
	inputs1 = list()
	inputs1.append(t1)
	inputs1.append(t2)
	inputs1.append(t3)
	return graph.concat(1, inputs1)




def inceptionE(graph, input):
	weight1 = graph.new_weight(dims=(320, input.dim(1), 1, 1))
	t1 = graph.conv2d(input=input, weight=weight1, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight2 = graph.new_weight(dims=(384, input.dim(1), 1, 1))
	t2 = graph.conv2d(input=input, weight=weight2, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight3 = graph.new_weight(dims=(384, 384, 1, 3))
	t2a = graph.conv2d(input=t2, weight=weight3, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight4 = graph.new_weight(dims=(384, 384, 3, 1))
	t2b = graph.conv2d(input=t2, weight=weight4, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs1 = list()
	inputs1.append(t2a)
	inputs1.append(t2b)
	t2 = graph.concat(1, inputs1)
	weight5 = graph.new_weight(dims=(448, input.dim(1), 1, 1))
	t3 = graph.conv2d(input=input, weight=weight5, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight6 = graph.new_weight(dims=(384, 448, 3, 3))
	t3 = graph.conv2d(input=t3, weight=weight6, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight7 = graph.new_weight(dims=(384, 384, 1, 3))
	t3a = graph.conv2d(input=t3, weight=weight7, strides=(1, 1),
                 padding="SAME", activation="RELU")
	weight8 = graph.new_weight(dims=(384, 384, 3, 1))
	t3b = graph.conv2d(input=t3, weight=weight8, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs2 = list()
	inputs2.append(t3a)
	inputs2.append(t3b)
	t3 = graph.concat(1, inputs2)
	t4 = graph.maxpool2d(input=input, kernels=(3,3), strides=(1, 1), padding="SAME")
	weight9 = graph.new_weight(dims=(192, input.dim(1), 1, 1))
	t4 = graph.conv2d(input=t4, weight=weight9, strides=(1, 1),
                 padding="SAME", activation="RELU")
	inputs3 = list()
	inputs3.append(t1)
	inputs3.append(t2)
	inputs3.append(t3)
	inputs3.append(t4)
	return graph.concat(1, inputs3)


# BUILD THE ORIGINAL GRAPH
graph = ts.new_graph()
input = graph.new_input(dims=(1,3,299,299))
weight = graph.new_weight(dims=(32,3,3,3))
t = graph.conv2d(input=input, weight=weight, strides=(2,2),
                 padding="VALID", activation="RELU")

weight1 = graph.new_weight(dims=(32,32,3,3))

t = graph.conv2d(input=t, weight=weight1, strides=(1,1),
                 padding="VALID", activation="RELU")

weight2 = graph.new_weight(dims=(64,32,3,3))

t = graph.conv2d(input=t, weight=weight2, strides=(1,1),
                 padding="SAME", activation="RELU")

t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="VALID")

weight3 = graph.new_weight(dims=(80,64,1,1))

t = graph.conv2d(input=t, weight=weight3, strides=(1,1),
                 padding="VALID", activation="RELU")

weight4 = graph.new_weight(dims=(192,80,3,3))

t = graph.conv2d(input=t, weight=weight4, strides=(1,1),
                 padding="VALID", activation="RELU")

t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="VALID")


t = inceptionA(graph, t, 192, 32)
t = inceptionA(graph, t, 256, 64)
t = inceptionA(graph, t, 288, 64)
t = inceptionB(graph, t)
t = inceptionC(graph, t, 128)
t = inceptionC(graph, t, 160)
t = inceptionC(graph, t, 160)
t = inceptionC(graph, t, 192)
t = inceptionD(graph, t)
t = inceptionE(graph, t)
t = inceptionE(graph, t)

t = graph.avgpool2d(input=t, kernels=(8, 8), strides=(1,1), padding="VALID")

#ts.export_to_file(graph, b"/usr/TASO/examples/inceptionv3.model")

old_time = graph.run_time()

#old_onnx = ts.export_onnx(graph)
#onnx.save(old_onnx, "inceptionv3-py.onnx")

new_graph = ts.optimize(graph, alpha=1.0, budget=100)

new_time = new_graph.run_time()

#new_onnx = ts.export_onnx(new_graph)
#onnx.save(new_onnx, "inceptionv3-py.taso.onnx")

print("Run time of original graph is: {}".format(old_time))
print("Run time of optimized graph is: {}".format(new_time))

args = get_args()
with open(args.result_file, "a") as f:
    f.write("{}\t{}\n".format(old_time, new_time))
