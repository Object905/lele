//! Graph rewrites on the [`ir`](super::ir).

mod fold;
mod qdq;
mod schedule;

pub(crate) use fold::fold_constants;
pub(crate) use qdq::{fuse_qdq, fuse_qdq_matmul};
pub(crate) use schedule::{group_dynamic_quant_matmul, sink_constant_of_shape};
