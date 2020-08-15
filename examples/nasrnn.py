import taso
import onnx

hidden_size = 512
length = 5

def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--result_file', type=str, default='nasrnn_time.txt', metavar='S',
        help='File to store times')
    return parser.parse_args()

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

old_time = graph.run_time()

new_graph = taso.optimize(graph, alpha=1.0, budget=100)

new_time = new_graph.run_time()
print("Run time of original graph is: {}".format(old_time))
print("Run time of optimized graph is: {}".format(new_time))

args = get_args()
with open(args.result_file, "a") as f:
    f.write("{}\t{}\n".format(old_time, new_time))