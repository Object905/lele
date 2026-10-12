use super::super::{generate::OpContext, ir::Op};
use std::io::Write;

pub(crate) fn handle_activation_ops(
    ctx: &mut OpContext,
    w: &mut dyn Write,
) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match op {
        "Relu" => writeln!(
            w,
            "{}let {} = lele::kernels::relu(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "LeakyRelu" => {
            let Op::LeakyRelu { alpha } = ctx.ir().op else {
                return Ok(false);
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::leaky_relu(&{}, {}, {});",
                tab, outputs[0], inputs[0], alpha, buf_expr
            )?;
        }
        "Sigmoid" => writeln!(
            w,
            "{}let {} = lele::kernels::sigmoid(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Softmax" | "LogSoftmax" => {
            let (Op::Softmax { axis } | Op::LogSoftmax { axis }) = ctx.ir().op else {
                return Ok(false);
            };
            // The opset 13+ default; before 13 it was 1.
            let axis = axis.unwrap_or(-1);
            let kernel = if op == "Softmax" {
                "softmax"
            } else {
                "log_softmax"
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::{}(&{}, {}, {});",
                tab, outputs[0], kernel, inputs[0], axis, buf_expr
            )?;
        }
        "ArgMax" => {
            let Op::ArgMax { axis, keepdims } = ctx.ir().op else {
                return Ok(false);
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::argmax(&{}, {}, {}, {});",
                tab, outputs[0], inputs[0], axis, keepdims as i64, buf_expr
            )?;
        }
        "Tanh" => writeln!(
            w,
            "{}let {} = lele::kernels::tanh_kernel(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "HardSigmoid" => {
            let Op::HardSigmoid { alpha, beta } = ctx.ir().op else {
                return Ok(false);
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::hard_sigmoid(&{}, {:?}, {:?}, {});",
                tab, outputs[0], inputs[0], alpha, beta, buf_expr
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
