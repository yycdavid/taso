import taso as ts
import onnx
import sys

graph = ts.new_graph()
guid_node = dict()

# can replace while true with walrus in python 3.8
while True:
  l = sys.stdin.readline()
  if not l:
    break
  guid = int(l.strip())
  op = int(sys.stdin.readline().strip())
  deps = [int(s.split(':')[0]) for s in sys.stdin.readline().strip().split(',')]
  params = [int(s) for s in sys.stdin.readline().strip().split(',')]

  node = {
    'Input': lambda: graph.new_input(dims=tuple(params)),
    'Weight': lambda: graph.new_weight(dims=tuple(params)),
    # activation for matmul
    'Matmul': lambda: graph.matmul(
                     guid_node[deps[0]],
                     guid_node[deps[1]]),
    'Reshape': lambda: graph.reshape(guid_node[deps[0]], shape=tuple(params)),
    # perm, shuffle
    'Transpose': lambda: graph.transpose(guid_node[deps[0]], perm=(1,0,2), shuffle=True),
    'Relu': lambda: graph.relu(guid_node[deps[0]]),
  }.get(ts.op_table[op], lambda: '')
  guid_node[guid] = node()

onnx_model = ts.export_onnx(graph)
onnx.save(onnx_model, "nasneta_taso.onnx")
