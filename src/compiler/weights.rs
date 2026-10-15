//! Layout of the weights file.

use super::generate::ElemType;
use super::ir::{DType, Graph, Layout, Op};
use super::sanitize_name;
use crate::model::onnx_proto::TensorProto;
use crate::model::{tensor_to_array, tensor_to_vec_u8};
use std::cell::RefCell;
use std::collections::HashMap;

/// Where a weight lives in the weights file. Only this module knows where:
/// code generation reads a weight through the expressions it builds.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WeightRef {
    /// Byte offset.
    offset: usize,
    /// Length in bytes.
    len: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// An f16 matrix stored as [`pack_f16`](crate::kernels::matmul::pack_f16)
    /// lays it out (or lays out its transpose: see
    /// [`Layout::PackedF16`]).
    pub packed: bool,
    /// The value of a rank-0 weight, which code reads inline.
    pub scalar: Option<Scalar>,
}

impl WeightRef {
    /// The call of `lele::runtime::Weights` method `accessor` reading this
    /// weight, on `self` in a model method.
    pub(crate) fn read(&self, accessor: &str) -> String {
        format!(
            "self.{accessor}({}, {}, &{:?})",
            self.offset, self.len, self.shape
        )
    }

    /// The byte offset and length of this weight, as the arguments the
    /// runtime methods reading raw bytes (`pack_qweights`, ...) start with.
    pub(crate) fn bytes(&self) -> String {
        format!("{}, {}", self.offset, self.len)
    }

    /// The expression reading this weight as `target`: a scalar inline, any
    /// other weight through the accessor widening its type to `target`.
    pub(crate) fn expr(&self, target: ElemType) -> String {
        let float = |v: f32| match v {
            _ if v.is_finite() => format!("{v:?}f32"),
            _ if v.is_nan() => "f32::NAN".to_string(),
            _ if v > 0.0 => "f32::INFINITY".to_string(),
            _ => "f32::NEG_INFINITY".to_string(),
        };
        match (self.scalar, target) {
            (Some(Scalar::Int(v)), ElemType::I64) => return format!("lele::scalar({v}i64)"),
            (Some(Scalar::Int(v)), ElemType::F32) => {
                return format!("lele::scalar({})", float(v as f32));
            }
            (Some(Scalar::Float(v)), ElemType::F32) => {
                return format!("lele::scalar({})", float(v));
            }
            _ => {}
        }
        let accessor = match target {
            ElemType::I64 => match self.dtype {
                DType::I32 => "weight_i32_i64",
                DType::Bool => "weight_bool_i64",
                _ => "weight_i64",
            },
            ElemType::F32 => match self.dtype {
                DType::U8 => "weight_u8",
                DType::I8 => "weight_i8",
                DType::I32 => "weight_i32_f32",
                DType::I64 => "weight_i64_f32",
                DType::F16 => "weight_f16",
                _ => "weight_f32",
            },
        };
        self.read(accessor)
    }
}

/// The value of a rank-0 weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scalar {
    /// Of an int64, int32, 8-bit or bool weight.
    Int(i64),
    /// Of an f32 or f16 weight.
    Float(f32),
}

/// The weights file, with where each weight landed in it.
#[derive(Default)]
pub(crate) struct Weights {
    pub data: Vec<u8>,
    /// Weights by sanitized name.
    lookup: HashMap<String, WeightRef>,
    /// Values of int64 and int32 weights by name, for code generation to
    /// inline.
    pub int64: HashMap<String, (Vec<i64>, Vec<usize>)>,
    /// Offset and length of each distinct content already written.
    by_content: HashMap<u64, (usize, usize)>,
    /// Weights of the integer products, packed when the model is created.
    pub prepared: RefCell<Prepared>,
}

/// Weights generated code prepares for the integer kernels when the model is
/// created, as the expressions preparing each, in the order code numbers them.
#[derive(Default)]
pub(crate) struct Prepared {
    pub qweights: Vec<String>,
    pub conv_weights: Vec<String>,
}

/// The number of `expr` in `prepared`, added if new.
fn number(prepared: &mut Vec<String>, expr: String) -> usize {
    prepared.iter().position(|e| *e == expr).unwrap_or_else(|| {
        prepared.push(expr);
        prepared.len() - 1
    })
}

impl Weights {
    /// The number of the [`QWeights`](crate::kernels::QWeights) that `expr`
    /// prepares from `weights` when the model is created.
    pub(crate) fn qweights(&self, expr: String) -> usize {
        number(&mut self.prepared.borrow_mut().qweights, expr)
    }

    /// The number of the [`ConvWeights`](crate::kernels::ConvWeights) that
    /// `expr` prepares from `weights` when the model is created.
    pub(crate) fn conv_weights(&self, expr: String) -> usize {
        number(&mut self.prepared.borrow_mut().conv_weights, expr)
    }

    /// Lays out the initializers and `Constant` values of `g` and its
    /// subgraphs: per graph, its initializers in model order, then each
    /// node's constant followed by its subgraphs' weights, in attribute
    /// order. Each weight
    /// starts 16-byte aligned; identical contents are stored once.
    pub(crate) fn collect(g: &Graph) -> Self {
        let mut weights = Weights::default();
        weights.add_graph(g);
        weights
    }

    /// The weight holding the tensor `name`.
    pub(crate) fn get(&self, name: &str) -> Option<&WeightRef> {
        self.lookup.get(&sanitize_name(name))
    }

    /// The values of `name`, if it is an int64 or int32 weight.
    pub(crate) fn ints(&self, name: &str) -> Option<&[i64]> {
        self.int64.get(name).map(|(ints, _)| ints.as_slice())
    }

    /// Byte offset and length of every weight of `dtype` stored as it is (not
    /// packed) that `code` reads through `accessor`, in file order, for the
    /// runtime to widen once at load.
    pub(crate) fn read_through(
        &self,
        dtype: DType,
        accessor: &str,
        code: &str,
    ) -> Vec<(usize, usize)> {
        let mut regions: Vec<_> = self
            .lookup
            .values()
            .filter(|w| w.dtype == dtype && !w.packed)
            .map(|w| (w.offset, w.len))
            .collect();
        regions.sort_unstable();
        regions.dedup();
        regions.retain(|(offset, len)| code.contains(&format!("self.{accessor}({offset}, {len},")));
        regions
    }

    /// Whether the tensor `name` is read from the weights rather than
    /// computed.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.get(name).is_some() || self.int64.contains_key(name)
    }

    fn add_graph(&mut self, g: &Graph) {
        for &v in g.initializers() {
            // Those that folding left without readers are not written.
            if let Some(t) = g.data(v).filter(|_| g.uses(v) > 0) {
                let value = g.value(v);
                self.add(&value.name, t, value.layout == Layout::PackedF16);
            }
        }
        for (_, node) in g.nodes() {
            if let Op::Constant { value } = &node.op
                && let Some(out) = node.output(0)
            {
                self.add(&g.value(out).name, value, false);
            }
            // In the order the node lists them.
            let mut subgraphs = node.op.named_subgraphs();
            subgraphs
                .sort_by_key(|(name, _)| node.proto.attribute.iter().position(|a| a.name == *name));
            for (_, sub) in subgraphs {
                self.add_graph(sub);
            }
        }
    }

    fn add(&mut self, name: &str, t: &TensorProto, packed: bool) {
        let dtype = DType::from_onnx(t.data_type);
        if matches!(dtype, DType::I64 | DType::I32)
            && let Ok((data, dims)) = tensor_to_array(t)
        {
            let ints = data.iter().map(|&x| x as i64).collect();
            self.int64.insert(name.to_string(), (ints, dims));
        }
        let Ok((bytes, shape, _)) = tensor_to_vec_u8(t) else {
            return;
        };
        if bytes.is_empty() {
            return;
        }
        let scalar = match dtype {
            _ if !shape.is_empty() => None,
            DType::I64 => bytes.first_chunk().map(|&b| Scalar::Int(i64::from_le_bytes(b))),
            DType::I32 => bytes.first_chunk().map(|&b| Scalar::Int(i32::from_le_bytes(b).into())),
            DType::Bool => Some(Scalar::Int((bytes[0] != 0).into())),
            DType::U8 => Some(Scalar::Int(bytes[0].into())),
            DType::I8 => Some(Scalar::Int((bytes[0] as i8).into())),
            DType::F32 => bytes.first_chunk().map(|&b| Scalar::Float(f32::from_le_bytes(b))),
            DType::F16 => bytes
                .first_chunk()
                .map(|&b| Scalar::Float(half::f16::from_le_bytes(b).to_f32())),
            _ => None,
        };
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut hasher);
            hasher.finish()
        };
        let (offset, len) = *self.by_content.entry(hash).or_insert_with(|| {
            self.data.resize(self.data.len().next_multiple_of(16), 0);
            let offset = self.data.len();
            self.data.extend_from_slice(&bytes);
            (offset, bytes.len())
        });
        // The first weight of a name wins.
        self.lookup.entry(sanitize_name(name)).or_insert(WeightRef {
            offset,
            len,
            shape,
            dtype,
            packed,
            scalar,
        });
    }
}
