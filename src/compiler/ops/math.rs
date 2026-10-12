use super::super::generate::{ElemType, OpContext};
use super::super::ir::{DType, Op, Reduction};
use super::super::sanitize_name;
use std::io::Write;

pub(crate) fn handle_math_ops(ctx: &mut OpContext, w: &mut dyn Write) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;
    let buf_expr = &ctx.buf_expr;

    match op {
        "Add" => writeln!(
            w,
            "{}let {} = lele::kernels::add(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "Sub" => writeln!(
            w,
            "{}let {} = lele::kernels::sub(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "Mul" => writeln!(
            w,
            "{}let {} = lele::kernels::mul(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "Div" => {
            // ONNX Div takes operands of one type, so an integer initializer
            // makes it an integer division. When it runs in f32 (e.g. on the
            // f32 indices of TopK) the quotient still has to be truncated.
            let integer_in_f32 = !ctx.var_types.is_i64(&outputs[0])
                && (0..ctx.node.input.len()).any(|i| {
                    ctx.weight(i)
                        .is_some_and(|w| matches!(w.dtype, DType::I32 | DType::I64))
                });
            let kernel = if integer_in_f32 { "div_trunc" } else { "div" };
            writeln!(
                w,
                "{}let {} = lele::kernels::{}(&{}, &{}, {});",
                tab, outputs[0], kernel, inputs[0], inputs[1], buf_expr
            )?
        }
        "Mod" => writeln!(
            w,
            "{}let {} = lele::kernels::mod_f32(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "MatMul" => writeln!(
            w,
            "{}let {} = lele::kernels::matmul(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "MatMulInteger" => {
            // MatMulInteger expects u8 inputs.
            let a_zp = if inputs.len() > 2 {
                format!("Some(&{})", inputs[2])
            } else {
                "None".to_string()
            };
            let b_zp = if inputs.len() > 3 {
                format!("Some(&{})", inputs[3])
            } else {
                "None".to_string()
            };
            // A static B is packed once for the integer GEMM.
            if let Some(b) = ctx.weight(1) {
                if matches!(b.dtype, DType::U8 | DType::I8) && b.shape.len() == 2 {
                    let k = b.shape[0];
                    let n = b.shape[1];
                    writeln!(
                        w,
                        "{}let {} = self.mat_mul_integer_packed(&{}, {}, {}, {}, {}, {}, {}, {}, {});",
                        tab,
                        outputs[0],
                        inputs[0],
                        b.offset,
                        b.len,
                        k,
                        n,
                        b.dtype == DType::I8,
                        a_zp,
                        b_zp,
                        buf_expr
                    )?;
                } else {
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});",
                        tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr
                    )?;
                }
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::mat_mul_integer(&{}, &{}, {}, {}, {});",
                    tab, outputs[0], inputs[0], inputs[1], a_zp, b_zp, buf_expr
                )?;
            }
        }
        "Pow" => writeln!(
            w,
            "{}let {} = lele::kernels::pow(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "Sqrt" => writeln!(
            w,
            "{}let {} = lele::kernels::sqrt(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Neg" => writeln!(
            w,
            "{}let {} = lele::kernels::neg(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Reciprocal" => writeln!(
            w,
            "{}let {} = lele::kernels::reciprocal(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Erf" => writeln!(
            w,
            "{}let {} = lele::kernels::erf(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Softplus" => writeln!(
            w,
            "{}let {} = lele::kernels::softplus(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Exp" => writeln!(
            w,
            "{}let {} = lele::kernels::exp(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Log" => writeln!(
            w,
            "{}let {} = lele::kernels::log(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Sin" => writeln!(
            w,
            "{}let {} = lele::kernels::sin(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Cos" => writeln!(
            w,
            "{}let {} = lele::kernels::cos(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Round" => writeln!(
            w,
            "{}let {} = lele::kernels::round(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Floor" => writeln!(
            w,
            "{}let {} = lele::kernels::floor(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Ceil" => writeln!(
            w,
            "{}let {} = lele::kernels::ceil(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "Equal" => {
            let types = ctx.var_types;
            if types.is_i64(&ctx.outputs[0]) || types.is_i64(&ctx.inputs[0]) {
                let a_is_weight_f32 = inputs[0].contains("weight_f32");
                let b_is_weight_f32 = inputs.len() > 1 && inputs[1].contains("weight_f32");
                let type_a = types.get(&sanitize_name(&ctx.node.input[0]));
                if b_is_weight_f32 {
                    let func = if type_a == ElemType::I64 {
                        "equal_i64_f32_r_i64"
                    } else {
                        "equal_i64_f32_r"
                    };
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::{}(&{}, &{}, {});",
                        tab, outputs[0], func, inputs[0], inputs[1], buf_expr
                    )?;
                } else if a_is_weight_f32
                    || type_a == ElemType::F32
                        && (inputs[1].contains("weight_i64")
                            || types.is_i64(&sanitize_name(&ctx.node.input[1])))
                {
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::equal_i64_f32_lhs(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?;
                } else {
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::equal_i64(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?;
                }
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::equal(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            }
        }
        "Less" => {
            let is_i64 = ctx.var_types.is_i64(&inputs[0]) || ctx.var_types.is_i64(&inputs[1]);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less_i64(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            }
        }
        "LessOrEqual" => {
            let is_i64 = ctx.var_types.is_i64(&inputs[0]) || ctx.var_types.is_i64(&inputs[1]);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less_or_equal_i64(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less_or_equal(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            }
        }
        "Not" => writeln!(
            w,
            "{}let {} = lele::kernels::not(&{}, {});",
            tab, outputs[0], inputs[0], buf_expr
        )?,
        "PRelu" => writeln!(
            w,
            "{}let {} = lele::kernels::prelu(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "ReduceSum" | "ReduceProd" | "ReduceMax" | "ReduceMean" | "ReduceL2" => {
            let Op::Reduce {
                reduction,
                axes,
                keepdims,
            } = &ctx.ir().op
            else {
                return Ok(false);
            };
            let attr_axes = format!("&{:?}", axes.as_deref().unwrap_or_default());
            let axes = match reduction {
                // These kernels take computed axes too.
                Reduction::Sum | Reduction::Prod | Reduction::Max => {
                    if ctx.node.input.len() > 1 && !ctx.node.input[1].is_empty() {
                        match ctx.ints(1) {
                            Some(ints) => format!("&{:?}", ints),
                            None => format!("&lele::kernels::to_i64_vec(&{})", inputs[1]),
                        }
                    } else {
                        attr_axes
                    }
                }
                Reduction::Mean | Reduction::L2 => match ctx.ints(1) {
                    Some(ints) if axes.as_ref().is_none_or(|a| a.is_empty()) => {
                        format!("&{:?}", ints)
                    }
                    _ => attr_axes,
                },
            };
            let kernel = match reduction {
                Reduction::L2 => "reduce_l2",
                Reduction::Max => "reduce_max",
                Reduction::Mean => "reduce_mean",
                Reduction::Prod => "reduce_prod",
                Reduction::Sum => "reduce_sum",
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::{}(&{}, {}, {}, {});",
                tab, outputs[0], kernel, inputs[0], axes, keepdims, buf_expr
            )?;
        }
        "Max" => {
            // ONNX Max: element-wise maximum of two or more inputs
            if inputs.len() == 2 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::max(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            } else {
                // Chain: max(a, max(b, max(c, d)))
                let mut result = inputs[0].clone();
                for (i, input) in inputs.iter().enumerate().skip(1) {
                    if i < inputs.len() - 1 {
                        let tmp = format!("max_tmp_{}_{}", outputs[0], i);
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::max(&{}, &{}, {});",
                            tab, tmp, result, input, buf_expr
                        )?;
                        result = tmp;
                    } else {
                        writeln!(
                            w,
                            "{}let {} = lele::kernels::max(&{}, &{}, {});",
                            tab, outputs[0], result, input, buf_expr
                        )?;
                    }
                }
            }
        }
        "Range" => {
            let is_i64 = inputs[..3].iter().any(|i| ctx.var_types.is_i64(i));
            if is_i64 {
                writeln!(w, "{}let mut buf_{} = Vec::<i64>::new();", tab, outputs[0])?;
                writeln!(
                    w,
                    "{}let {} = lele::kernels::range_i64(&{}, &{}, &{}, &mut buf_{});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2], outputs[0]
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::range(&{}, &{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], inputs[2], buf_expr
                )?;
            }
        }
        "Clip" => {
            let min = if inputs.len() > 1 && !inputs[1].is_empty() {
                format!("Some(&{})", inputs[1])
            } else {
                "None".to_string()
            };
            let max = if inputs.len() > 2 && !inputs[2].is_empty() {
                format!("Some(&{})", inputs[2])
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::clip(&{}, {}, {}, {});",
                tab, outputs[0], inputs[0], min, max, buf_expr
            )?;
        }
        "Greater" => {
            let is_i64 = ctx.var_types.is_i64(&inputs[0]) || ctx.var_types.is_i64(&inputs[1]);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less_i64(&{}, &{}, {});",
                    tab, outputs[0], inputs[1], inputs[0], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::less(&{}, &{}, {});",
                    tab, outputs[0], inputs[1], inputs[0], buf_expr
                )?;
            }
        }
        "STFT" => {
            // ONNX STFT: inputs[0]=signal, inputs[1]=frame_step, inputs[2]=window, inputs[3]=frame_length
            let frame_step = ctx.ints(1).map_or(160, |ints| ints[0] as usize);
            let n_fft = ctx.ints(3).map_or(512, |ints| ints[0] as usize);
            let window = if ctx.node.input.len() > 2 && !ctx.node.input[2].is_empty() {
                if let Some(window) = ctx.weight(2) {
                    let (o, l, s) = (window.offset, window.len, &window.shape);
                    format!("Some(&self.weight_f32({}, {}, &{:?}))", o, l, s)
                } else {
                    format!("Some(&{})", inputs[2])
                }
            } else {
                "None".to_string()
            };
            writeln!(
                w,
                "{}let {} = lele::kernels::stft(&{}, {}, {}, {}, {}, {});",
                tab, outputs[0], inputs[0], n_fft, frame_step, n_fft, window, buf_expr
            )?;
        }
        "And" => writeln!(
            w,
            "{}let {} = lele::kernels::logical_and(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "Or" => writeln!(
            w,
            "{}let {} = lele::kernels::logical_or(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "Xor" => writeln!(
            w,
            "{}let {} = lele::kernels::logical_xor(&{}, &{}, {});",
            tab, outputs[0], inputs[0], inputs[1], buf_expr
        )?,
        "GreaterOrEqual" => {
            let is_i64 = ctx.var_types.is_i64(&inputs[0]) || ctx.var_types.is_i64(&inputs[1]);
            if is_i64 {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::greater_or_equal_i64(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            } else {
                writeln!(
                    w,
                    "{}let {} = lele::kernels::greater_or_equal(&{}, &{}, {});",
                    tab, outputs[0], inputs[0], inputs[1], buf_expr
                )?;
            }
        }
        "CumSum" => {
            let Op::CumSum { exclusive, reverse } = ctx.ir().op else {
                return Ok(false);
            };
            let axis = ctx.ints(1).map_or(0, |ints| ints[0]);
            writeln!(
                w,
                "{}let {} = lele::kernels::cumsum(&{}, {}, {}, {}, {});",
                tab, outputs[0], inputs[0], axis, exclusive, reverse, buf_expr
            )?;
        }
        "Einsum" => {
            let Op::Einsum { equation } = &ctx.ir().op else {
                return Ok(false);
            };
            let normalized = equation.replace(' ', "");
            match normalized.as_str() {
                "bs,d->bsd" => {
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::einsum_bs_d_bsd(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?;
                }
                other => {
                    writeln!(w, "{}// Unsupported Einsum: {}", tab, other)?;
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::einsum_bs_d_bsd(&{}, &{}, {});",
                        tab, outputs[0], inputs[0], inputs[1], buf_expr
                    )?;
                }
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
