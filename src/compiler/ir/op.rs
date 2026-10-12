//! Typed operators of the IR.

use super::Graph;
use crate::model::onnx_proto::{AttributeProto, NodeProto, TensorProto};

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
        set_attr_i(attrs, "axis", self.axis, 1);
        set_attr_i(attrs, "block_size", self.block_size, 0);
        set_attr_opt_i(
            attrs,
            "output_dtype",
            self.output_dtype.map(|t| t.to_onnx() as i64),
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
    Cast {
        to: DType,
    },
    Concat {
        axis: i64,
    },
    /// Only the `value` form; `value_float`, `value_ints` etc. stay
    /// [`Op::Onnx`].
    Constant {
        value: TensorProto,
    },
    ConstantOfShape {
        /// Fill value; zero f32 when absent.
        value: Option<TensorProto>,
    },
    MatMul,
    QuantizeLinear(QuantAttrs),
    DequantizeLinear(QuantAttrs),
    Shape {
        start: Option<i64>,
        end: Option<i64>,
    },
    /// Opset 10+ form, all parameters are operands. The opset 1 attributes
    /// stay in the proto.
    Slice,
    Squeeze {
        /// Opset 1-12 attribute; later opsets take the axes as an operand.
        axes: Option<Vec<i64>>,
    },
    Unsqueeze {
        /// Opset 1-12 attribute; later opsets take the axes as an operand.
        axes: Option<Vec<i64>>,
    },
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

    /// Any operator without a typed variant yet. Its graph attributes are
    /// built into `graphs` so passes see into them; every other attribute
    /// stays in the proto.
    Onnx {
        op_type: String,
        domain: String,
        graphs: Vec<(String, Graph)>,
    },
}

impl Op {
    /// Parses `node`'s operator. `subgraph` hands over the graph attribute of
    /// the given name, already built. Tensor attributes a variant takes are
    /// moved out of `node`.
    pub(super) fn parse(
        node: &mut NodeProto,
        mut subgraph: impl FnMut(&str) -> Option<Graph>,
    ) -> Self {
        let fallback = |node: &NodeProto| Op::Onnx {
            op_type: node.op_type.clone(),
            domain: node.domain.clone(),
            graphs: Vec::new(),
        };
        if !(node.domain.is_empty() || node.domain == "ai.onnx") {
            return fallback(node);
        }
        match node.op_type.as_str() {
            "Cast" => match attr_i(node, "to") {
                Some(to) => Op::Cast {
                    to: DType::from_onnx(to as i32),
                },
                None => fallback(node),
            },
            "Concat" => Op::Concat {
                axis: attr_i(node, "axis").unwrap_or(0),
            },
            "Constant" => match take_attr_t(node, "value") {
                Some(value) => Op::Constant { value },
                None => fallback(node),
            },
            "ConstantOfShape" => Op::ConstantOfShape {
                value: take_attr_t(node, "value"),
            },
            "MatMul" => Op::MatMul,
            "QuantizeLinear" => Op::QuantizeLinear(QuantAttrs::parse(node)),
            "DequantizeLinear" => Op::DequantizeLinear(QuantAttrs::parse(node)),
            "Shape" => Op::Shape {
                start: attr_i(node, "start"),
                end: attr_i(node, "end"),
            },
            "Slice" => Op::Slice,
            "Squeeze" => Op::Squeeze {
                axes: attr_ints(node, "axes"),
            },
            "Unsqueeze" => Op::Unsqueeze {
                axes: attr_ints(node, "axes"),
            },
            "If" => match (subgraph("then_branch"), subgraph("else_branch")) {
                (Some(t), Some(e)) => Op::If {
                    then_branch: Box::new(t),
                    else_branch: Box::new(e),
                },
                _ => fallback(node),
            },
            "Loop" => match subgraph("body") {
                Some(body) => Op::Loop {
                    body: Box::new(body),
                },
                None => fallback(node),
            },
            _ => fallback(node),
        }
    }

    /// The `op_type` the node lowers to.
    pub fn op_type(&self) -> &str {
        match self {
            Op::Cast { .. } => "Cast",
            Op::Concat { .. } => "Concat",
            Op::Constant { .. } => "Constant",
            Op::ConstantOfShape { .. } => "ConstantOfShape",
            Op::MatMul => "MatMul",
            Op::QuantizeLinear(_) => "QuantizeLinear",
            Op::DequantizeLinear(_) => "DequantizeLinear",
            Op::Shape { .. } => "Shape",
            Op::Slice => "Slice",
            Op::Squeeze { .. } => "Squeeze",
            Op::Unsqueeze { .. } => "Unsqueeze",
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
            Op::Onnx { graphs, .. } => graphs.iter().map(|(_, g)| g).collect(),
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
            Op::Onnx { graphs, .. } => graphs.iter_mut().map(|(_, g)| g).collect(),
            _ => Vec::new(),
        }
    }

    /// Writes the attributes this operator models into `attrs`, lowering its
    /// subgraphs.
    pub(super) fn into_attrs(self, attrs: &mut Vec<AttributeProto>) {
        match self {
            Op::Cast { to } => set_attr_opt_i(attrs, "to", Some(to.to_onnx() as i64)),
            Op::Concat { axis } => set_attr_i(attrs, "axis", axis, 0),
            Op::Constant { value } => set_attr_t(attrs, "value", Some(value)),
            Op::ConstantOfShape { value } => set_attr_t(attrs, "value", value),
            Op::QuantizeLinear(a) | Op::DequantizeLinear(a) | Op::FakeQuantizeLinear(a) => {
                a.write(attrs)
            }
            Op::Shape { start, end } => {
                set_attr_opt_i(attrs, "start", start);
                set_attr_opt_i(attrs, "end", end);
            }
            Op::Squeeze { axes } | Op::Unsqueeze { axes } => set_attr_ints(attrs, "axes", axes),
            Op::If {
                then_branch,
                else_branch,
            } => {
                set_attr_g(attrs, "then_branch", *then_branch);
                set_attr_g(attrs, "else_branch", *else_branch);
            }
            Op::Loop { body } => set_attr_g(attrs, "body", *body),
            Op::Onnx { graphs, .. } => {
                for (name, g) in graphs {
                    set_attr_g(attrs, &name, g);
                }
            }
            Op::MatMul | Op::Slice | Op::QLinearMatMulI8 => {}
        }
    }
}

fn attr_i(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

fn attr_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

fn take_attr_t(node: &mut NodeProto, name: &str) -> Option<TensorProto> {
    node.attribute
        .iter_mut()
        .find(|a| a.name == name)
        .and_then(|a| a.t.take())
}

/// Sets attribute `name` to `value`, adding it if absent unless it would
/// only restate `default`, so that an unchanged node lowers to exactly the
/// proto it was parsed from.
fn set_attr_i(attrs: &mut Vec<AttributeProto>, name: &str, value: i64, default: i64) {
    match attrs.iter_mut().find(|a| a.name == name) {
        Some(a) => a.i = value,
        None if value != default => attrs.push(AttributeProto {
            name: name.to_string(),
            r#type: 2, // INT
            i: value,
            ..Default::default()
        }),
        None => {}
    }
}

/// Sets attribute `name` to `value`, or removes it if `value` is `None`.
fn set_attr_opt_i(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<i64>) {
    match value {
        Some(v) => set_attr(attrs, name, 2 /* INT */, |a| a.i = v),
        None => attrs.retain(|a| a.name != name),
    }
}

fn set_attr_ints(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<Vec<i64>>) {
    match value {
        Some(v) => set_attr(attrs, name, 7 /* INTS */, |a| a.ints = v),
        None => attrs.retain(|a| a.name != name),
    }
}

fn set_attr_t(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<TensorProto>) {
    match value {
        Some(t) => set_attr(attrs, name, 4 /* TENSOR */, |a| a.t = Some(t)),
        None => attrs.retain(|a| a.name != name),
    }
}

fn set_attr_g(attrs: &mut Vec<AttributeProto>, name: &str, graph: Graph) {
    let g = graph.into_onnx();
    set_attr(attrs, name, 5 /* GRAPH */, |a| a.g = Some(g));
}

/// Updates attribute `name` with `set`, adding it with type `ty` if absent.
fn set_attr(
    attrs: &mut Vec<AttributeProto>,
    name: &str,
    ty: i32,
    set: impl FnOnce(&mut AttributeProto),
) {
    match attrs.iter_mut().find(|a| a.name == name) {
        Some(a) => set(a),
        None => {
            let mut a = AttributeProto {
                name: name.to_string(),
                r#type: ty,
                ..Default::default()
            };
            set(&mut a);
            attrs.push(a);
        }
    }
}
