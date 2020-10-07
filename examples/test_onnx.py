import taso
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to input ONNX file", required=True)

args = parser.parse_args()

graph = taso.load_onnx(args.file)

old_time = graph.run_time()

new_graph = taso.optimize(graph, alpha = 1.0, budget = 100, print_subst = True)

new_time = new_graph.run_time()
print("Run time of original graph is: {}".format(old_time))
print("Run time of optimized graph is: {}".format(new_time))

onnx_model = taso.export_onnx(new_graph)
onnx.save(onnx_model, "{}.taso.onnx".format(args.file))
