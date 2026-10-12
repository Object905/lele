//! Rescheduling of nodes for code generation.

use crate::compiler::ir::{Graph, NodeId, Op, ValueId};
use std::collections::{HashMap, HashSet};

/// Moves a `ConstantOfShape` whose shape is a compile-time constant to just
/// before the `Concat` that consumes it, so its buffer is not held across
/// the nodes in between.
pub(crate) fn sink_constant_of_shape(g: &mut Graph) {
    let order = g.node_ids();
    let pos: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let is_const = |v: ValueId| {
        g.data(v).is_some()
            || g.producer(v)
                .is_some_and(|p| matches!(p.op, Op::Constant { .. }))
    };

    // Concat position -> producers to move in front of it.
    let mut moves: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut moved: HashSet<usize> = HashSet::new();
    for (i, &id) in order.iter().enumerate() {
        let node = g.node(id);
        if !matches!(node.op, Op::Concat { .. }) || node.inputs().len() < 2 {
            continue;
        }
        for &v in node.inputs().iter().flatten() {
            let Some(p_id) = g.value(v).producer() else {
                continue;
            };
            let p = pos[&p_id];
            let producer = g.node(p_id);
            if p + 1 < i
                && producer.output(0) == Some(v)
                && matches!(producer.op, Op::ConstantOfShape { .. })
                && producer.input(0).is_some_and(is_const)
                // Feeding several Concats, it goes before the first.
                && moved.insert(p)
            {
                moves.entry(i).or_default().push(p);
            }
        }
    }
    if moves.is_empty() {
        return;
    }

    let mut new_order = Vec::with_capacity(order.len());
    for (i, &id) in order.iter().enumerate() {
        for &p in moves.get(&i).into_iter().flatten() {
            new_order.push(order[p]);
        }
        if !moved.contains(&i) {
            new_order.push(id);
        }
    }
    g.set_order(new_order);
}

/// Makes the dynamically quantized matmul chain
/// `DynamicQuantizeLinear → Mul(scales) → MatMulInteger → Cast → Mul → Add`
/// contiguous, so that its code generation pattern, which matches a run of
/// consecutive nodes, finds it.
///
/// The chain is searched for within 30 nodes of the `DynamicQuantizeLinear`.
/// Unrelated nodes interleaved with it are moved in front of it, which is
/// only done if none of them reads a chain output.
pub(crate) fn group_dynamic_quant_matmul(g: &mut Graph) {
    let order = g.node_ids();
    let op = |i: usize| g.node(order[i]).op.op_type();
    let node = |i: usize| g.node(order[i]);
    let reads = |i: usize, v: ValueId| node(i).inputs().contains(&Some(v));

    let mut chains: Vec<[usize; 6]> = Vec::new();
    let mut in_chain: HashSet<usize> = HashSet::new();
    for i in 0..order.len() {
        if op(i) != "DynamicQuantizeLinear" || in_chain.contains(&i) {
            continue;
        }
        let (Some(q), Some(s), Some(z)) = (node(i).output(0), node(i).output(1), node(i).output(2))
        else {
            continue;
        };
        let window = (i + 30).min(order.len());

        // Mul(scale, weight_scale) → combined scale.
        let Some(mul_scales) = (i + 1..window).find(|&j| op(j) == "Mul" && reads(j, s)) else {
            continue;
        };
        // MatMulInteger(quantized, weight, zero point).
        let Some(mmi) = (i + 1..window).find(|&j| {
            let n = node(j);
            op(j) == "MatMulInteger"
                && n.inputs().len() >= 3
                && n.input(0) == Some(q)
                && n.input(2) == Some(z)
        }) else {
            continue;
        };
        let (Some(mm_out), Some(combined_scale)) =
            (node(mmi).output(0), node(mul_scales).output(0))
        else {
            continue;
        };
        let Some(cast) =
            (mmi + 1..window).find(|&j| op(j) == "Cast" && node(j).input(0) == Some(mm_out))
        else {
            continue;
        };
        let Some(cast_out) = node(cast).output(0) else {
            continue;
        };
        let Some(mul_dq) = (cast + 1..window)
            .find(|&j| op(j) == "Mul" && reads(j, cast_out) && reads(j, combined_scale))
        else {
            continue;
        };
        let Some(dequant) = node(mul_dq).output(0) else {
            continue;
        };
        let Some(add) = (mul_dq + 1..window).find(|&j| op(j) == "Add" && reads(j, dequant)) else {
            continue;
        };

        let chain = [i, mul_scales, mmi, cast, mul_dq, add];
        let chain_set: HashSet<usize> = chain.iter().copied().collect();
        let chain_outputs: HashSet<ValueId> = chain
            .iter()
            .flat_map(|&j| node(j).outputs().iter().flatten().copied())
            .collect();
        let safe = (i..=add).all(|j| {
            chain_set.contains(&j)
                || !node(j)
                    .inputs()
                    .iter()
                    .flatten()
                    .any(|v| chain_outputs.contains(v))
        });
        if safe {
            chains.push(chain);
            in_chain.extend(chain);
        }
    }
    if chains.is_empty() {
        return;
    }

    let mut new_order = Vec::with_capacity(order.len());
    let mut emitted = vec![false; order.len()];
    for i in 0..order.len() {
        if emitted[i] {
            continue;
        }
        let Some(chain) = chains.iter().find(|c| c[0] == i) else {
            new_order.push(order[i]);
            emitted[i] = true;
            continue;
        };
        // Interleaved nodes first, in their order, then the chain.
        for j in i..=chain[5] {
            if !chain.contains(&j) && !emitted[j] {
                new_order.push(order[j]);
                emitted[j] = true;
            }
        }
        for &j in chain {
            new_order.push(order[j]);
            emitted[j] = true;
        }
    }
    g.set_order(new_order);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::{GraphProto, NodeProto, TensorProto, ValueInfoProto};

    fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn graph(nodes: Vec<NodeProto>, initializers: &[&str]) -> GraphProto {
        let last = nodes.last().unwrap().output[0].clone();
        GraphProto {
            node: nodes,
            initializer: initializers
                .iter()
                .map(|name| TensorProto {
                    name: name.to_string(),
                    data_type: 7,
                    dims: vec![1],
                    int64_data: vec![2],
                    ..Default::default()
                })
                .collect(),
            output: vec![ValueInfoProto {
                name: last,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Output names of the nodes in execution order.
    fn run(g: GraphProto, pass: fn(&mut Graph)) -> Vec<String> {
        let mut ir = Graph::from_onnx(g);
        pass(&mut ir);
        ir.into_onnx()
            .node
            .into_iter()
            .map(|n| n.output[0].clone())
            .collect()
    }

    #[test]
    fn sinks_constant_of_shape_to_its_concat() {
        let order = run(
            graph(
                vec![
                    node("ConstantOfShape", &["shape"], &["c"]),
                    node("Relu", &["x"], &["a"]),
                    node("Relu", &["a"], &["b"]),
                    node("Concat", &["b", "c"], &["y"]),
                ],
                &["shape"],
            ),
            sink_constant_of_shape,
        );
        assert_eq!(order, ["a", "b", "c", "y"]);
    }

    #[test]
    fn keeps_constant_of_shape_with_a_dynamic_shape() {
        let order = run(
            graph(
                vec![
                    node("Shape", &["x"], &["shape"]),
                    node("ConstantOfShape", &["shape"], &["c"]),
                    node("Relu", &["x"], &["a"]),
                    node("Relu", &["a"], &["b"]),
                    node("Concat", &["b", "c"], &["y"]),
                ],
                &[],
            ),
            sink_constant_of_shape,
        );
        assert_eq!(order, ["shape", "c", "a", "b", "y"]);
    }

    /// The dynamically quantized matmul chain with `other` interleaved.
    fn dynamic_quant_matmul(other: NodeProto) -> GraphProto {
        graph(
            vec![
                node("DynamicQuantizeLinear", &["x"], &["q", "s", "z"]),
                other,
                node("Mul", &["s", "w_scale"], &["scale"]),
                node("MatMulInteger", &["q", "w", "z", "w_zp"], &["mm"]),
                node("Cast", &["mm"], &["mm_f"]),
                node("Mul", &["mm_f", "scale"], &["dq"]),
                node("Add", &["bias", "dq"], &["y"]),
            ],
            &[],
        )
    }

    #[test]
    fn groups_dynamic_quant_matmul_chain() {
        let order = run(
            dynamic_quant_matmul(node("Relu", &["x"], &["other"])),
            group_dynamic_quant_matmul,
        );
        assert_eq!(order, ["other", "q", "scale", "mm", "mm_f", "dq", "y"]);
    }

    #[test]
    fn keeps_chain_when_an_interleaved_node_reads_it() {
        // Hoisting `other` would move it before `s` is computed.
        let order = run(
            dynamic_quant_matmul(node("Relu", &["s"], &["other"])),
            group_dynamic_quant_matmul,
        );
        assert_eq!(order, ["q", "other", "scale", "mm", "mm_f", "dq", "y"]);
    }
}
