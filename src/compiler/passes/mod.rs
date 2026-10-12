//! Graph rewrites on the [`ir`](super::ir).

mod qdq;

pub(crate) use qdq::{fuse_qdq, fuse_qdq_matmul};
