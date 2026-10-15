//! Graph rewrites on the [`ir`](super::ir).

mod branches;
mod cse;
mod dce;
mod fold;
mod fuse;
mod qdq;
mod shapes;
mod simplify;
mod types;

pub(crate) use branches::inline_constant_ifs;
pub(crate) use cse::eliminate_common_subexpressions;
pub(crate) use dce::eliminate_dead_code;
pub(crate) use fold::fold_constants;
pub(crate) use fuse::fuse;
pub(crate) use qdq::{
    ACT_REQUANT, I8_BIAS, I8_REQUANT, fold_qdq_into_matmuls, fuse_qdq, fuse_qdq_matmul,
    fuse_weight_only_matmul,
};
pub(crate) use shapes::fold_shapes;
pub(crate) use simplify::{bypass_view_casts, simplify};
pub(crate) use types::{follows_input, infer_dtypes};

use super::ir::{Graph, Node, Op};

/// The index of a `Gather` of one constant index along the outermost axis,
/// whose result is one slab of its input: code reads it as a view of that
/// slab ([`select`](crate::kernels::select)), with no buffer of its own.
pub(crate) fn selected_index(g: &Graph, node: &Node) -> Option<i64> {
    if !matches!(node.op, Op::Gather { axis: 0 }) {
        return None;
    }
    let index = node.input(1)?;
    let t = match g.producer(index) {
        Some(Node {
            op: Op::Constant { value },
            ..
        }) => value,
        _ => g.data(index)?,
    };
    match (t.dims.is_empty(), fold::exact_ints(t)?.as_slice()) {
        (true, &[i]) => Some(i),
        _ => None,
    }
}
