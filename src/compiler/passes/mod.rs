//! Graph rewrites on the [`ir`](super::ir).

mod fold;
mod qdq;

pub(crate) use fold::fold_constants;
pub(crate) use qdq::{fuse_qdq, fuse_qdq_matmul};
