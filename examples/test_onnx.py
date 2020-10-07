import taso
import onnx
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to input ONNX file", required=True)
parser.add_argument('--result_file', type=str, default='new_time.txt', metavar='S',
    help='File to store times')
parser.add_argument('--alpha', type=float, default=1.0,
    help='Threshold value')
parser.add_argument('--iter', type=int, default=100,
    help='Number of iterations for backtracking search')

args = parser.parse_args()

graph = taso.load_onnx(args.file)

old_time = graph.run_time()

new_graph = taso.optimize(graph, alpha = args.alpha, budget = args.iter)

new_time = new_graph.run_time()
print("Run time of original graph is: {}".format(old_time))
print("Run time of optimized graph is: {}".format(new_time))

with open(args.result_file, "a") as f:
    f.write("{}\t{}\n".format(old_time, new_time))

#onnx_model = taso.export_onnx(new_graph)
#onnx.save(onnx_model, "{}.taso.onnx".format(args.file))
