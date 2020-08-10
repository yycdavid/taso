#! /usr/bin/python

import sys
import os
from itertools import product

import z3
from z3 import ForAll

import re


T = z3.DeclareSort('T')
P = z3.IntSort()

OP_INPUT, \
OP_WEIGHT, \
OP_ANY, \
OP_CONV2D, \
OP_DROPOUT, \
OP_LINEAR, \
OP_POOL2D_MAX, \
OP_POOL2D_AVG, \
OP_RELU, \
OP_SIGMOID, \
OP_TANH, \
OP_BATCHNORM, \
OP_CONCAT, \
OP_SPLIT, \
OP_RESHAPE, \
OP_TRANSPOSE, \
OP_EW_ADD, \
OP_EW_MUL, \
OP_MATMUL, \
OP_MUL, \
OP_ENLARGE, \
OP_MERGE_GCONV, \
OP_CONSTANT_IMM, \
OP_CONSTANT_ICONV, \
OP_CONSTANT_ONE, \
OP_CONSTANT_POOL = range(26)

PM_OP_TYPE, \
PM_NUM_INPUTS, \
PM_NUM_OUTPUTS, \
PM_GROUP, \
PM_KERNEL_H, \
PM_KERNEL_W, \
PM_STRIDE_H, \
PM_STRIDE_W, \
PM_PAD, \
PM_ACTI, \
PM_NUMDIM, \
PM_AXIS, \
PM_PERM, \
PM_OUTSHUFFLE, \
PM_MERGE_GCONV_COUNT = range(15)

AC_MODE_NONE, \
AC_MODE_SIGMOID, \
AC_MODE_RELU, \
AC_MODE_TANH = range(4)

PD_MODE_SAME, \
PD_MODE_VALID = range(2)

# map opId to (name, (key,rng)*, input arity, outputa arity, possible input dimensions)
operator_data = {
    OP_CONSTANT_POOL: ('Cpool', ((PM_KERNEL_H, {3}), (PM_KERNEL_W, {3})), 0, 1, {}),
    OP_CONSTANT_ICONV: ('Iconv', ((PM_KERNEL_H, {3}), (PM_KERNEL_W, {3})), 0, 1, {}),
    OP_CONSTANT_IMM: ('Imatmul', (), 0, 1, {}),
    OP_CONSTANT_ONE: ('Iewmul', (), 0, 1, {}),
    OP_CONV2D: ('conv2d', ((PM_STRIDE_H, {1,2}), (PM_STRIDE_W, {1,2}), (PM_PAD, {0,1}), (PM_ACTI, {AC_MODE_NONE, AC_MODE_RELU})), 2, 1, {4}),
    OP_POOL2D_MAX: ('poolmax', ((PM_KERNEL_H, {3}), (PM_KERNEL_W, {3}), (PM_STRIDE_H, {1, 2}), (PM_STRIDE_W, {1,2}), (PM_PAD, {0,1})), 1, 1, {4}),
    OP_POOL2D_AVG: ('poolavg', ((PM_KERNEL_H, {3}), (PM_KERNEL_W, {3}), (PM_STRIDE_H, {1, 2}), (PM_STRIDE_W, {1,2}), (PM_PAD, {0,1})), 1, 1, {4}),
    OP_RELU: ('relu', (), 1, 1, {2, 3, 4}),
    OP_CONCAT: ('concat', ((PM_AXIS, {0, 1, 2, 3}), (PM_NUMDIM, {2,4}),), 2, 1, {2,3,4}),
    OP_SPLIT: ('split', ((PM_AXIS, {0, 1, 2, 3}),), 1, 2, {2,3,4}),
    OP_TRANSPOSE: ('transpose', (), 1, 1, {2}),
    OP_ENLARGE: ('enlarge', ((PM_KERNEL_H, {3}), (PM_KERNEL_W, {3})), 1, 1, {4}),
    OP_EW_ADD: ('ewadd', (), 2, 1, {2,3,4}),
    OP_EW_MUL: ('ewmul', (), 2, 1, {2,3,4}),
    OP_MATMUL: ('matmul', ((PM_ACTI, {AC_MODE_NONE, AC_MODE_RELU}),), 2, 1, {2}),
    OP_MUL: ('smul', (), 2, 1, {2, 3, 4}) # multiply a tensor (first argument) with a scalar (0-D tensor)
}

for d in operator_data.values():
    for i in range(d[3]):
        name = d[0]
        globals()[name] = z3.Function(name, *( len(d[1]) * [P] + d[2] * [T] + [T]))

name = "split_0"
globals()[name] = z3.Function(name, *(1 * [T] + [T]))
name = "split_1"
globals()[name] = z3.Function(name, *(1 * [T] + [T]))

x,y,z,w, one = z3.Consts('x y z w one', T)
sx, sy, kx, ky, pad, acti, ax = z3.Consts('sx sy kx ky pad acti ax', P)

N = [1,2,3,4] # change this to control number of combinations for symbolic validation, e.g., [1,2], [1,3] or [3,4] each provide a reasonable experiment to run and go for coffee (assuming 8 cores)
D = [1,3]

# list of axioms with list of possible values for verify_axioms.py. possible values are actual values for parameters, and shapes for tensors

#    (ForAll([kx, ky, x, y], enlarge_0(kx, ky, ewadd_0(x, y)) == ewadd_0(enlarge_0(kx, ky, x), enlarge_0(kx, ky, y))),
#     lambda : [(kx, ky, s, s)
#               for kx, ky in product(D, repeat=2)
#               for s in product(N, repeat=4)
#     ]),
#
#    (ForAll([kx, ky, x, y], enlarge_0(kx, ky, ewmul_0(x, y)) == ewmul_0(enlarge_0(kx, ky, x), enlarge_0(kx, ky, y))),
#     lambda : [(kx, ky, s, s)
#               for kx, ky in product(D, repeat=2)
#               for s in product(N, repeat=4)
#     ]),
#
#    (ForAll([kx, ky, x, w], enlarge_0(kx, ky, scalar_mul_0(x, w)) == scalar_mul_0(enlarge_0(kx, ky, x), w)),
#     lambda : [(kx, ky, s, ())
#               for kx, ky in product(D, repeat=2)
#               for s in product(N, repeat=4)
#     ]),
#
#    (ForAll([kx, ky, x, y], enlarge_0(kx, ky, concat_0(0, x, y)) == concat_0(0, enlarge_0(kx, ky, x), enlarge_0(kx, ky, y))),
#     lambda : [(kx, ky, s1, s2)
#               for kx, ky in product(D, repeat=2)
#               for s1 in product(N, repeat=4)
#               for s2 in product(N, repeat=4)
#               if all(s1[i] == s2[i] or i == 0 for i in range(4))
#     ]),
#
#    (ForAll([kx, ky, x, y], enlarge_0(kx, ky, concat_0(1, x, y)) == concat_0(1, enlarge_0(kx, ky, x), enlarge_0(kx, ky, y))),
#     lambda : [(kx, ky, s1, s2)
#               for kx, ky in product(D, repeat=2)
#               for s1 in product(N, repeat=4)
#               for s2 in product(N, repeat=4)
#               if all(s1[i] == s2[i] or i == 1 for i in range(4))
#     ]),
#
#    (ForAll([kx, ky, x], enlarge_0(kx, ky, relu_0(x)) == relu_0(enlarge_0(kx, ky, x))),
#     lambda : [(kx, ky, s)
#               for kx, ky in product(D, repeat=2)
#               for s in product(N, repeat=4)
#     ]),
#
    # concat is associative (wrong axiom - makes many others redundant)
    # (ForAll([ax, x, y, z], concat_0(ax, x, concat_0(ax, y,z)) == concat_0(ax, concat_0(ax, x, y), z)),
    #  lambda : [(ax, s1, s2, s3)
    #            for dim in [2,3,4]
    #            for s1 in product(N, repeat=dim)
    #            for s2 in product(N, repeat=dim)
    #            for s3 in product(N, repeat=dim)
    #            for ax in range(dim)
    #            if all(s1[i] == s2[i] == s3[i] or i == ax for i in range(dim))
    #  ]),

    # grouped convolution (wrong axiom - caught with N=[1,3])
    # (ForAll([sx, sy, pad, acti, x, y, z, w], concat_0(1, conv2d_0(sx, sy, pad, acti, x, y), conv2d_0(sx, sy, pad, acti, z, w)) == conv2d_0(sx, sy, pad, acti, concat_0(1, x, z), concat_0(0, y, w))),
    #  lambda :[(sx, sy, pad, acti, (n,cx,h,w), (c1y,c2,d1,d2), (n,cz,h,w), (c1w,c2,d1,d2))
    #           for sx in [1,2]
    #           for sy in [1,2]
    #           for pad in [PD_MODE_SAME, PD_MODE_VALID]
    #           for acti in [AC_MODE_NONE, AC_MODE_RELU]
    #           for n,cx,h,w,c1y,c2,cz,c1w in product(N,repeat=8)
    #           for d1 in D
    #           for d2 in D
    #           if all([
    #                   h >= d1,
    #                   w >= d2,
    #                   cx % c2 == 0,
    #                   cz % c2 == 0,
    #                   (cx // c2) > 0 and c1y % (cx // c2) == 0,
    #                   (cz // c2) > 0 and c1w % (cz // c2) == 0,
    #                   ((cx + cz) // c2) > 0 and (c1w + c1y) % ((cx + cz) // c2) == 0,
    #           ])
    #  ]),

ops_present = set()

def to_z3(tensor, ops):
    if tensor.opId < 0:
        # an input tensor
        return z3.Const('input_{}'.format(-tensor.opId), T)
    else:
        op = ops[tensor.opId]
        d = operator_data[op.type]
        if not d[0] in ops_present:
            print(d[0])
            ops_present.add(d[0])
        
        #print(op.type, d)
        assert tensor.tsId <= d[3]
        f = globals()['{}'.format(d[0])]
        params = {}
        for p in op.para:
            params[p.key] = p.value
        args = []

        for k, rng in d[1]:
            v = params[k]
            if (v not in rng):
                print(k, v, rng)
                assert False
            assert v in rng
            assert type(v) is int
            args.append(v)
        args += [to_z3(x, ops) for x in op.input]

        if d[0] == "split":
            f_out = globals()['{}_{}'.format(d[0], tensor.tsId)]
            return f_out(*[f(*args)])
        else:
            return f(*args)

def check_bounded(src, dst):
    src_expr = "{}".format(src)
    dst_expr = "{}".format(dst)

    src_inputs = find_all_inputs(src_expr)
    dst_inputs = find_all_inputs(dst_expr)
    for i in dst_inputs:
        if not i in src_inputs:
            return False

    return True

def find_all_inputs(expr):
    start_ids = [m.start() for m in re.finditer('input', expr)]
    all_inputs = set()
    for i in start_ids:
        all_inputs.add(expr[i:i+7])

    return all_inputs


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "<graph substitutions file>")
        sys.exit(-1)

    import rules_pb2
    rules = rules_pb2.RuleCollection()
    rules.ParseFromString(open(sys.argv[1], "rb").read())

    # print("Axioms:\n{}".format([a for a, b in axioms]))

    '''blacklist = {
        # some substitutions that are known to be incorrect and should be skipped
        #'nasnet_subst.pb': [166, 167, 186, 187, 222, 223, 224, 225, 226, 227, 283, 284, 290, 291, 298, 299],
        #'graph_subst.pb': [178, 179, 387, 405, 429, 443, 444, 485, 486, 487, 488, 489, 490, 548, 549, 555, 556, 563, 564],
        #'graph_subst.pb': [201, 202, 209, 247, 259, 264, 265, 316, 527, 528, 529, 532, 573, 584, 585, 586, 607, 627, 628, 670, 671, 672, 673, 674, 675, 740, 741, 751, 752, 761, 762],
        #'graph_subst.pb': [202, 209, 254, 255, 260, 307, 308, 518, 536, 560, 580, 581, 620, 621, 622, 623, 624, 625, 681, 682, 688, 689, 695, 696, 697],
        'new_graph_subst.pb': [],
    }[os.path.basename(sys.argv[1])]'''
    blacklist = []

    print("Number of rules: {}\n".format(len(rules.rule)))
    print("Number of rules after blacklist: {}\n".format(len(rules.rule) - len(blacklist)))

    convert_rules = True

    if convert_rules:
        cnt = 0
        unbounded_cnt = 0
        multi_cnt = 0
        with open("multi_rules.txt", "w") as fout:
            with open("blk_rules.txt", "w") as fout_b:
                for i, rule in enumerate(rules.rule):
                    if i in blacklist:
                        for output in rule.mappedOutput:
                            # print("Verifing output: {}".format(output))
                            src_tensor = rules_pb2.Tensor(opId=output.srcOpId, tsId=output.srcTsId)
                            dst_tensor = rules_pb2.Tensor(opId=output.dstOpId, tsId=output.dstTsId)
                            src = to_z3(src_tensor, rule.srcOp)
                            dst = to_z3(dst_tensor, rule.dstOp)

                            rule_str = "{}".format(src == dst)
                            rule_str = rule_str.replace(' ','').replace('\n', '').replace('\r', '').replace('\t', '')
                            fout_b.write(rule_str+'\n')
                        continue
                    # print("Verifying rule: {} with {} outputs\n".format(rule, len(rule.mappedOutput)))
                    if len(rule.mappedOutput) > 1:
                        multi_cnt += len(rule.mappedOutput)
                        print(len(rule.mappedOutput))

                        for output in rule.mappedOutput:
                            # print("Verifing output: {}".format(output))
                            src_tensor = rules_pb2.Tensor(opId=output.srcOpId, tsId=output.srcTsId)
                            dst_tensor = rules_pb2.Tensor(opId=output.dstOpId, tsId=output.dstTsId)
                            src = to_z3(src_tensor, rule.srcOp)
                            dst = to_z3(dst_tensor, rule.dstOp)

                            rule_str = "{}".format(src == dst)
                            rule_str = rule_str.replace(' ','').replace('\n', '').replace('\r', '').replace('\t', '')
                            fout.write(rule_str)
                            #fout.write('\n')
                        continue

                    '''
                    for output in rule.mappedOutput:
                        # print("Verifing output: {}".format(output))
                        src_tensor = rules_pb2.Tensor(opId=output.srcOpId, tsId=output.srcTsId)
                        dst_tensor = rules_pb2.Tensor(opId=output.dstOpId, tsId=output.dstTsId)
                        src = to_z3(src_tensor, rule.srcOp)
                        dst = to_z3(dst_tensor, rule.dstOp)
                        if not check_bounded(src, dst):
                            unbounded_cnt += 1

                        rule_str = "{}".format(src == dst)
                        #import pdb; pdb.set_trace()
                        rule_str = rule_str.replace(' ','').replace('\n', '').replace('\r', '').replace('\t', '')
                        fout.write(rule_str)
                        cnt += 1
                    '''

        print("Number of rules single: {}\n".format(cnt))
        print("Number of rules with multi: {}\n".format(multi_cnt + cnt))
        print("number of unbounded: {}".format(unbounded_cnt))

    else:
        cnt = 0
        multi_cnt = 0

        idx_to_rm = []
        for i, rule in enumerate(rules.rule):
            if i in blacklist:
                idx_to_rm.append(i)
            elif len(rule.mappedOutput) > 1:
                idx_to_rm.append(i)
                multi_cnt += len(rule.mappedOutput)
            else:
                cnt += 1

        print("Number of rules single: {}\n".format(cnt))
        print("Number of rules with multi: {}\n".format(multi_cnt + cnt))

        for i in idx_to_rm[::-1]:
            del rules.rule[i]

        print("Number of rules remains: {}\n".format(len(rules.rule)))

        with open("graph_subst_single.pb", "wb") as fd:
            fd.write(rules.SerializeToString())
