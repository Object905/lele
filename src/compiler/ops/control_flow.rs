use super::super::{
    generate::{OpContext, generate_nodes, weight_expr},
    sanitize_name,
};
use std::io::Write;

pub(crate) fn handle_control_flow_ops(
    ctx: &mut OpContext,
    w: &mut dyn Write,
) -> std::io::Result<bool> {
    let op = ctx.node.op_type.as_str();
    let tab = ctx.tab();
    let inputs = &ctx.inputs;
    let outputs = &ctx.outputs;

    match op {
        "If" => {
            let then_branch = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "then_branch")
                .unwrap()
                .g
                .as_ref()
                .unwrap();
            let else_branch = ctx
                .node
                .attribute
                .iter()
                .find(|a| a.name == "else_branch")
                .unwrap()
                .g
                .as_ref()
                .unwrap();
            // A branch output is either a weight or a variable of the branch.
            let (known_weights, var_types) = (ctx.known_weights, ctx.var_types);
            let branch_output = |o: &crate::model::onnx_proto::ValueInfoProto| {
                let name = sanitize_name(&o.name);
                match known_weights.get(&name) {
                    Some((offset, len, shape, data_type)) => {
                        let ty = var_types.get(&name).map_or("f32", |s| s.as_str());
                        let weight = weight_expr(*offset, *len, shape, *data_type, ty);
                        format!("{weight}.to_owned()")
                    }
                    None => format!("{name}.to_owned()"),
                }
            };
            let cond = &inputs[0];
            let out_vars = outputs.join(", ");
            let cond_is_i64 = ctx.var_types.get(cond).map(|t| t == "i64").unwrap_or(false);
            if cond_is_i64 {
                writeln!(
                    w,
                    "{}let ({}) = if {}.data.first().is_some_and(|&v| v != 0) {{",
                    tab, out_vars, cond
                )?;
            } else {
                writeln!(
                    w,
                    "{}let ({}) = if {}.data.first().is_some_and(|&v| v != 0.0) {{",
                    tab, out_vars, cond
                )?;
            }

            let then_nodes: Vec<&crate::model::onnx_proto::NodeProto> =
                then_branch.node.iter().collect();
            generate_nodes(
                &then_nodes,
                w,
                ctx.indent + 1,
                ctx.known_weights,
                ctx.int64_map,
                ctx.allocator,
                None, // Subgraphs have their own scope
                ctx.current_id,
                ctx.compiler,
                ctx.var_types,
            )?;

            let then_outs: Vec<String> = then_branch.output.iter().map(branch_output).collect();
            writeln!(w, "{}    ({})", tab, then_outs.join(", "))?;
            writeln!(w, "{}}} else {{", tab)?;

            let else_nodes: Vec<&crate::model::onnx_proto::NodeProto> =
                else_branch.node.iter().collect();
            generate_nodes(
                &else_nodes,
                w,
                ctx.indent + 1,
                ctx.known_weights,
                ctx.int64_map,
                ctx.allocator,
                None,
                ctx.current_id,
                ctx.compiler,
                ctx.var_types,
            )?;

            let else_outs: Vec<String> = else_branch.output.iter().map(branch_output).collect();
            writeln!(w, "{}    ({})", tab, else_outs.join(", "))?;
            writeln!(w, "{}}};", tab)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
