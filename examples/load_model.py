import taso as ts
import onnx
import sys

# convert serialized taso graph to onnx graph
# to run: python load_model.py < your.model
# output is stored in `out.onnx`

ac_mode = {
  ts.get_activation_mode("NONE"): "NONE",
  ts.get_activation_mode("SIGMOID"): "SIGMOID",
  ts.get_activation_mode("RELU"): "RELU",
  ts.get_activation_mode("TANH"): "TANH",
}

padding_mode = {
  ts.get_padding_mode("SAME"): "SAME",
  ts.get_padding_mode("VALID"): "VALID",
}

graph = ts.new_graph()
guid_node = dict()

# can replace while true with walrus in python 3.8
while True:
  print(3)
  l = sys.stdin.readline()
  if not l:
    break
  guid = int(l.strip())
  op = int(sys.stdin.readline().strip())
  deps = [(int(s.split(':')[0]), int(s.split(':')[1])) for s in sys.stdin.readline().strip().split(',')]
  params = [int(s) for s in sys.stdin.readline().strip().split(',')]

  print(op)
  print(ts.op_table[op])
  node = {
    'Input':     lambda: [graph.new_input(dims=tuple(params))],
    'Weight':    lambda: [graph.new_weight(dims=tuple(params))],
    'Matmul':    lambda: [graph.matmul(guid_node[deps[0][0]][deps[0][1]], guid_node[deps[1][0]][deps[1][1]], activation=ac_mode[params[0]])],
    'MaxPool':   lambda: [graph.maxpool2d(guid_node[deps[0][0]][deps[0][1]], kernels = (params[5], params[6]), strides = (params[7], params[8]), padding = padding_mode[params[9]], activation=ac_mode[params[10]])],
    'AveragePool':   lambda: [graph.avgpool2d(input = guid_node[deps[0][0]][deps[0][1]], kernels = (params[5], params[6]), strides = (params[7], params[8]), padding = padding_mode[params[9]], activation=ac_mode[params[10]])],
    'Enlarge':   lambda: [graph.enlarge(guid_node[deps[0][0]][deps[0][1]], guid_node[deps[1][0]][deps[1][1]])],
    'Add':       lambda: [graph.add(guid_node[deps[0][0]][deps[0][1]], guid_node[deps[1][0]][deps[1][1]])],
    'Reshape':   lambda: [graph.reshape(guid_node[deps[0][0]][deps[0][1]], shape=tuple(params))],
    'MergeGConv':lambda: [graph.merge_gconv(guid_node[deps[0][0]][deps[0][1]], count=params[0])],
    'Transpose': lambda: [graph.transpose(guid_node[deps[0][0]][deps[0][1]], perm=tuple(params[:3]), shuffle=params[3])],
    'Relu':      lambda: [graph.relu(guid_node[deps[0][0]][deps[0][1]])],
    'Split':     lambda: graph.split(guid_node[deps[0][0]][deps[0][1]], axis=params[0], sizes=params[1:]),
    'Concat':    lambda: [graph.concat(axis = params[0], inputs = [guid_node[dep[0]][dep[1]] for dep in deps])],
    'Conv':      lambda: [graph.conv2d(
                           input = guid_node[deps[0][0]][deps[0][1]],
                           weight = guid_node[deps[1][0]][deps[1][1]],
                           strides = (params[8], params[9]),
                           padding = padding_mode[params[10]],
                           activation = ac_mode[params[11]])],
  }[ts.op_table[op]]
  guid_node[guid] = node()

onnx_model = ts.export_onnx(graph)
onnx.save(onnx_model, "out.onnx")
