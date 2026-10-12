//! Typed operators of the IR.

use super::Graph;
use crate::model::onnx_proto::{AttributeProto, NodeProto};

/// ONNX tensor element type (`TensorProto.DataType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DType {
    #[default]
    Undefined,
    F32,
    U8,
    I8,
    U16,
    I16,
    I32,
    I64,
    String,
    Bool,
    F16,
    F64,
    U32,
    U64,
    BF16,
    /// Any other `TensorProto.DataType` (complex, float8, int4, ...).
    Other(i32),
}

impl DType {
    pub fn from_onnx(data_type: i32) -> Self {
        match data_type {
            0 => Self::Undefined,
            1 => Self::F32,
            2 => Self::U8,
            3 => Self::I8,
            4 => Self::U16,
            5 => Self::I16,
            6 => Self::I32,
            7 => Self::I64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::F16,
            11 => Self::F64,
            12 => Self::U32,
            13 => Self::U64,
            16 => Self::BF16,
            other => Self::Other(other),
        }
    }

    pub fn to_onnx(self) -> i32 {
        match self {
            Self::Undefined => 0,
            Self::F32 => 1,
            Self::U8 => 2,
            Self::I8 => 3,
            Self::U16 => 4,
            Self::I16 => 5,
            Self::I32 => 6,
            Self::I64 => 7,
            Self::String => 8,
            Self::Bool => 9,
            Self::F16 => 10,
            Self::F64 => 11,
            Self::U32 => 12,
            Self::U64 => 13,
            Self::BF16 => 16,
            Self::Other(other) => other,
        }
    }
}

/// Attributes shared by `QuantizeLinear`, `DequantizeLinear` and the fused
/// `FakeQuantizeLinear`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantAttrs {
    pub axis: i64,
    pub block_size: i64,
    /// Explicit `output_dtype` (opset 21+). When absent, `QuantizeLinear`
    /// takes its type from the zero point, or defaults to uint8.
    pub output_dtype: Option<DType>,
}

impl QuantAttrs {
    fn parse(node: &NodeProto) -> Self {
        Self {
            axis: attr_i(node, "axis").unwrap_or(1),
            block_size: attr_i(node, "block_size").unwrap_or(0),
            output_dtype: attr_i(node, "output_dtype").map(|t| DType::from_onnx(t as i32)),
        }
    }

    fn write(&self, attrs: &mut Vec<AttributeProto>) {
        set_attr_i(attrs, "axis", Some(self.axis), 1);
        set_attr_i(attrs, "block_size", Some(self.block_size), 0);
        set_attr_i(
            attrs,
            "output_dtype",
            self.output_dtype.map(|t| t.to_onnx() as i64),
            0,
        );
    }
}

/// An operator together with its parsed attributes.
///
/// A typed variant owns the attributes it models: they are parsed once when
/// the graph is built and written back to the node's [`NodeProto`] when it is
/// lowered. Attributes a variant does not model, and every attribute of an
/// [`Op::Onnx`] node, stay in the proto untouched.
#[derive(Clone, Debug)]
pub enum Op {
    // Standard ONNX operators.
    MatMul,
    QuantizeLinear(QuantAttrs),
    DequantizeLinear(QuantAttrs),
    If {
        then_branch: Box<Graph>,
        else_branch: Box<Graph>,
    },
    Loop {
        body: Box<Graph>,
    },

    // lele operators, created by passes.
    /// `QuantizeLinear` → `DequantizeLinear` round trip in one step. Inputs
    /// are those of the `QuantizeLinear`.
    FakeQuantizeLinear(QuantAttrs),
    /// `MatMul` of a uint8-quantized activation against a symmetric int8
    /// weight. Inputs: `[a, a_scale, a_zero_point, w, w_scale]`.
    QLinearMatMulI8,

    /// Any operator without a typed variant yet.
    Onnx {
        op_type: String,
        domain: String,
    },
}

impl Op {
    /// Parses `node`'s operator. `subgraph` hands over the graph attribute of
    /// the given name, already built; graph attributes a variant does not
    /// take stay in the proto.
    pub(super) fn parse(node: &NodeProto, mut subgraph: impl FnMut(&str) -> Option<Graph>) -> Self {
        let onnx = node.domain.is_empty() || node.domain == "ai.onnx";
        let fallback = || Op::Onnx {
            op_type: node.op_type.clone(),
            domain: node.domain.clone(),
        };
        if !onnx {
            return fallback();
        }
        match node.op_type.as_str() {
            "MatMul" => Op::MatMul,
            "QuantizeLinear" => Op::QuantizeLinear(QuantAttrs::parse(node)),
            "DequantizeLinear" => Op::DequantizeLinear(QuantAttrs::parse(node)),
            "If" => match (subgraph("then_branch"), subgraph("else_branch")) {
                (Some(t), Some(e)) => Op::If {
                    then_branch: Box::new(t),
                    else_branch: Box::new(e),
                },
                _ => fallback(),
            },
            "Loop" => match subgraph("body") {
                Some(body) => Op::Loop {
                    body: Box::new(body),
                },
                None => fallback(),
            },
            _ => fallback(),
        }
    }

    /// The `op_type` the node lowers to.
    pub fn op_type(&self) -> &str {
        match self {
            Op::MatMul => "MatMul",
            Op::QuantizeLinear(_) => "QuantizeLinear",
            Op::DequantizeLinear(_) => "DequantizeLinear",
            Op::If { .. } => "If",
            Op::Loop { .. } => "Loop",
            Op::FakeQuantizeLinear(_) => "FakeQuantizeLinear",
            Op::QLinearMatMulI8 => "QLinearMatMulI8",
            Op::Onnx { op_type, .. } => op_type,
        }
    }

    /// Subgraphs owned by this operator.
    pub fn subgraphs(&self) -> Vec<&Graph> {
        match self {
            Op::If {
                then_branch,
                else_branch,
            } => vec![then_branch, else_branch],
            Op::Loop { body } => vec![body],
            _ => Vec::new(),
        }
    }

    pub fn subgraphs_mut(&mut self) -> Vec<&mut Graph> {
        match self {
            Op::If {
                then_branch,
                else_branch,
            } => vec![then_branch, else_branch],
            Op::Loop { body } => vec![body],
            _ => Vec::new(),
        }
    }

    /// Writes the attributes this operator models into `attrs`, lowering its
    /// subgraphs.
    pub(super) fn into_attrs(self, attrs: &mut Vec<AttributeProto>) {
        match self {
            Op::QuantizeLinear(a) | Op::DequantizeLinear(a) | Op::FakeQuantizeLinear(a) => {
                a.write(attrs)
            }
            Op::If {
                then_branch,
                else_branch,
            } => {
                set_attr_g(attrs, "then_branch", *then_branch);
                set_attr_g(attrs, "else_branch", *else_branch);
            }
            Op::Loop { body } => set_attr_g(attrs, "body", *body),
            Op::MatMul | Op::QLinearMatMulI8 | Op::Onnx { .. } => {}
        }
    }
}

fn attr_i(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

/// Sets integer attribute `name` to `value`. An attribute that is absent and
/// would only restate `default` is left out, so that an unchanged node lowers
/// to exactly the proto it was parsed from.
fn set_attr_i(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<i64>, default: i64) {
    let pos = attrs.iter().position(|a| a.name == name);
    match (pos, value) {
        (Some(pos), Some(v)) => attrs[pos].i = v,
        (Some(pos), None) => {
            attrs.remove(pos);
        }
        (None, Some(v)) if v != default => attrs.push(AttributeProto {
            name: name.to_string(),
            r#type: 2, // INT
            i: v,
            ..Default::default()
        }),
        (None, _) => {}
    }
}

fn set_attr_g(attrs: &mut Vec<AttributeProto>, name: &str, graph: Graph) {
    let g = graph.into_onnx();
    match attrs.iter_mut().find(|a| a.name == name) {
        Some(a) => a.g = Some(g),
        None => attrs.push(AttributeProto {
            name: name.to_string(),
            r#type: 5, // GRAPH
            g: Some(g),
            ..Default::default()
        }),
    }
}
