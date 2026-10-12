//! Code generation for the operators the fusion passes create.

use super::super::{generate::OpContext, generate::weight_expr, sanitize_name};
use crate::model::onnx_proto::NodeProto;
use std::io::Write;

/// A Conv node's `strides`, `dilations`, `group`, `pads` and `auto_pad`.
/// Strides and dilations come out with one entry per spatial axis of a 2D
/// convolution (a single value covers both); each axis keeps its own, so a
/// stride of `[2, 1]` stays `[2, 1]`.
fn conv_attributes(conv: &NodeProto) -> (Vec<i64>, Vec<i64>, usize, Vec<i64>, String) {
    let per_axis = |ints: &[i64]| match ints {
        [] => vec![1, 1],
        [v] => vec![*v, *v],
        _ => ints.to_vec(),
    };
    let (mut strides, mut dilations) = (vec![1, 1], vec![1, 1]);
    let (mut groups, mut pads, mut auto_pad) = (1, Vec::new(), String::new());
    for attr in &conv.attribute {
        match attr.name.as_str() {
            "strides" => strides = per_axis(&attr.ints),
            "dilations" => dilations = per_axis(&attr.ints),
            "group" => groups = attr.i as usize,
            "pads" => pads = attr.ints.clone(),
            "auto_pad" => auto_pad = String::from_utf8_lossy(&attr.s).to_string(),
            _ => {}
        }
    }
    (strides, dilations, groups, pads, auto_pad)
}

fn attr<'n>(
    node: &'n NodeProto,
    name: &str,
) -> Option<&'n crate::model::onnx_proto::AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

pub(crate) fn handle_fused_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let tab = ctx.tab();
    let (inputs, outputs, buf) = (&ctx.inputs, &ctx.outputs, &ctx.buf_expr);
    let node = ctx.node;
    match node.op_type.as_str() {
        "Silu" => writeln!(
            w,
            "{tab}let {} = lele::kernels::silu(&{}, {buf});",
            outputs[0], inputs[0]
        )?,
        "Gelu" => {
            let kernel = match attr(node, "approximate").map(|a| a.s.as_slice()) {
                None | Some(b"none") => "gelu_erf",
                Some(b"tanh") => "fast_gelu",
                Some(other) => panic!(
                    "Gelu approximate={:?} is not supported",
                    String::from_utf8_lossy(other)
                ),
            };
            writeln!(
                w,
                "{tab}let {} = lele::kernels::{kernel}(&{}, {buf});",
                outputs[0], inputs[0]
            )?;
        }
        "Linear" => writeln!(
            w,
            "{tab}let {} = self.linear(&{}, &{}, &{}, {buf});",
            outputs[0], inputs[0], inputs[1], inputs[2]
        )?,
        "ConvAct" => {
            let weight_shape = ctx
                .known_weights
                .get(&sanitize_name(&node.input[1]))
                .map(|(_, _, shape, _)| shape.clone())
                .unwrap_or(vec![3, 3, 3, 3]);
            let bias = match node.input.get(2) {
                Some(b) if !b.is_empty() => format!("Some(&{})", inputs[2]),
                _ => "None".to_string(),
            };
            let act = attr(node, "activation").map(|a| a.s.as_slice());
            if act == Some(b"Silu") {
                // Attributes as written, with ONNX defaults.
                let ints = |name: &str, default: &[i64]| {
                    attr(node, name).map_or(default.to_vec(), |a| a.ints.clone())
                };
                let group = attr(node, "group").map_or(1, |a| a.i);
                writeln!(
                    w,
                    "{tab}let {} = lele::kernels::conv2d_silu(&{}, &{}, {bias}, &{:?}, {group}, &{:?}, &{:?}, {buf});",
                    outputs[0],
                    inputs[0],
                    inputs[1],
                    ints("dilations", &[1, 1]),
                    ints("pads", &[0, 0, 0, 0]),
                    ints("strides", &[1, 1]),
                )?;
                return Ok(true);
            }
            let (strides, dilations, groups, mut pads, auto_pad) = conv_attributes(node);
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") && pads.is_empty() {
                let kh = weight_shape.get(2).copied().unwrap_or(1) as i64;
                let kw = weight_shape.get(3).copied().unwrap_or(kh as usize) as i64;
                let total_h = (kh - 1) * dilations[0];
                let total_w = (kw - 1) * dilations[1];
                let split = |total: i64| {
                    if auto_pad == "SAME_LOWER" {
                        (total - total / 2, total / 2)
                    } else {
                        (total / 2, total - total / 2)
                    }
                };
                let ((pt, pb), (pl, pr)) = (split(total_h), split(total_w));
                pads = vec![pt, pl, pb, pr];
            }
            if pads.is_empty() {
                pads = vec![0, 0, 0, 0];
            }
            if weight_shape.len() >= 4 {
                writeln!(
                    w,
                    "{tab}let {} = lele::kernels::conv2d_fused(&{}, &{}, {bias}, &{:?}, {groups}, &{:?}, &{:?}, true, {buf});",
                    outputs[0], inputs[0], inputs[1], dilations, pads, strides,
                )?;
            } else {
                writeln!(
                    w,
                    "{tab}let {} = self.conv1d_relu({}, {}, {bias}, {}, {}, {groups}, {}, {buf});",
                    outputs[0], inputs[0], inputs[1], strides[0], dilations[0], pads[0] as usize,
                )?;
            }
        }
        "QLinearDynamic" => {
            let relu = attr(node, "relu").is_some_and(|a| a.i != 0);
            let (x, scale, zero_point, bias) = (&inputs[0], &inputs[2], &inputs[3], &inputs[4]);
            // A static weight is packed once for the integer GEMM.
            match ctx.known_weights.get(&sanitize_name(&node.input[1])) {
                Some((offset, len, shape, dt)) if (*dt == 2 || *dt == 3) && shape.len() == 2 => {
                    writeln!(
                        w,
                        "{tab}let {} = self.linear_quantized_packed(&{x}, {offset}, {len}, {}, {}, {}, {scale}, {zero_point}, {bias}, {relu}, {buf});",
                        outputs[0],
                        shape[0],
                        shape[1],
                        *dt == 3,
                    )?
                }
                _ => {
                    let method = if relu {
                        "linear_quantized_relu"
                    } else {
                        "linear_quantized"
                    };
                    writeln!(
                        w,
                        "{tab}let {} = self.{method}(&{x}, {}, {scale}, {zero_point}, {bias}, {buf});",
                        outputs[0], inputs[1],
                    )?
                }
            }
        }
        "EmbeddingConcat" => {
            let value = attr(node, "value").and_then(|a| a.t.as_ref());
            let int_value = value.and_then(|t| {
                let int = t.data_type == 6
                    || t.data_type == 7
                    || !t.int64_data.is_empty()
                    || !t.int32_data.is_empty();
                int.then(|| {
                    t.int64_data
                        .first()
                        .copied()
                        .or(t.int32_data.first().map(|&v| v as i64))
                        .unwrap_or(0)
                })
            });
            // Weights are read as the type of the fill value.
            let operand = |i: usize, ty: &str| {
                let name = sanitize_name(&node.input[i]);
                match ctx.known_weights.get(&name) {
                    Some((offset, len, shape, dt)) => weight_expr(*offset, *len, shape, *dt, ty),
                    None => name,
                }
            };
            let shape = operand(0, "i64");
            match int_value {
                Some(v) => writeln!(
                    w,
                    "{tab}let {} = self.embedding_concat_i64(&{shape}, {v}, {}, {buf});",
                    outputs[0],
                    operand(1, "i64"),
                )?,
                None => {
                    let v = value
                        .and_then(|t| t.float_data.first())
                        .copied()
                        .unwrap_or(0.0);
                    writeln!(
                        w,
                        "{tab}let {} = self.embedding_concat(&{shape}, {v:?}, {}, {buf});",
                        outputs[0],
                        operand(1, "f32"),
                    )?;
                }
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
