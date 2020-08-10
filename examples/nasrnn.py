import taso
import onnx

hidden_size = 512
length = 5

def combine(graph, x, h):
    w1 = graph.new_input(dims=(hidden_size, x.dim(1)))
    w2 = graph.new_input(dims=(hidden_size, h.dim(1)))
    return graph.add(graph.matmul(x, w1), graph.matmul(h, w2))

def nas_node(graph, input, x):
    t = list()
    for i in range(8):
        t.append(combine(graph, x, input))
    midt = list()
    midt.append(graph.add(graph.relu(t[0]), graph.sigmoid(t[3])))
    midt.append(graph.add(graph.sigmoid(t[1]), graph.tanh(t[2])))
    midt.append(graph.mul(graph.sigmoid(t[4]), graph.tanh(t[5])))
    midt.append(graph.mul(graph.sigmoid(t[6]), graph.relu(t[7])))
    midt.append(graph.add(graph.sigmoid(midt[1]), graph.tanh(midt[2])))
    midt.append(graph.mul(graph.tanh(midt[0]), graph.tanh(midt[3])))
    midt.append(graph.mul(graph.tanh(midt[4]), graph.tanh(midt[5])))
    return graph.tanh(midt[6])

graph = taso.new_graph()
xs = list()
for i in range(length):
    xs.append(graph.new_input(dims=(1, hidden_size)))
state = graph.new_input(dims=(1, hidden_size))
for i in range(length):
    state = nas_node(graph, state, xs[i])

new_graph = taso.optimize(graph, alpha=1.0, budget=100)

graph = graph.preprocess_weights()
old_time = graph.run_time()
old_onnx = taso.export_onnx(graph)
onnx.save(old_onnx, "old_nasrnn_pre.onnx")

print("Run time of original graph is: {}".format(old_time))

new_time = new_graph.run_time()
print("Run time of optimized graph is: {}".format(new_time))
onnx_model = taso.export_onnx(new_graph)

onnx.save(onnx_model, "new_nasrnn.onnx")
