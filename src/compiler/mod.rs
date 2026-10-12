use crate::model::onnx_proto::{GraphProto, NodeProto};
use crate::model::tensor_to_array;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::Write;
/// Handler callback for custom operator code generation.
/// Args:
/// - node: The ONNX NodeProto
/// - inputs: List of Rust variable names for inputs (e.g. `["op_1_out", "state_in"]`)
/// - outputs: List of Rust variable names for outputs (e.g. `["op_5_out"]`)
/// - buffer: The code string to access the output buffer (e.g. `"&mut ws.buf_3"` or `"&mut buf_X"`)
/// - writer: The output writer
/// - indent: Current indentation level
pub type OpHandler = Box<
    dyn Fn(&NodeProto, &[String], &[String], &str, &mut dyn Write, usize) -> std::io::Result<()>,
>;
pub type PatternMatcher = Box<dyn Fn(&[&NodeProto]) -> Option<usize>>;
pub type PatternGenerator = Box<
    dyn Fn(
        &[&NodeProto],
        &HashMap<String, (usize, usize, Vec<usize>, i32)>,
        Option<&Allocator>,
        &mut dyn Write,
        usize,
    ) -> std::io::Result<()>,
>;
pub struct Pattern {
    pub name: String,
    pub matcher: PatternMatcher,
    pub generator: PatternGenerator,
}
pub struct CompilationResult {
    pub code: String,
    pub weights: Vec<u8>,
}
pub struct Allocator {
    // Maps tensor name -> Buffer Index
    pub tensor_to_buffer: HashMap<String, usize>,
    pub num_buffers: usize,
}
// Helper to identify nodes uniquely across recursion
struct NodeVisitor<'a> {
    id_counter: usize,
    // Map tensor name -> List of Node IDs that use it
    uses: HashMap<String, Vec<usize>>,
    // Map Node ID -> NodeProto
    node_map: HashMap<usize, &'a NodeProto>,
    // Map Node ID -> outputs
    node_outputs: HashMap<usize, Vec<String>>,
    // Map Node ID -> inputs
    node_inputs: HashMap<usize, Vec<String>>,
    // Map alias -> sources (e.g. If_out -> [Then_out, Else_out])
    aliases: HashMap<String, Vec<String>>,
}
impl<'a> NodeVisitor<'a> {
    fn new() -> Self {
        Self {
            id_counter: 0,
            uses: HashMap::new(),
            node_map: HashMap::new(),
            node_outputs: HashMap::new(),
            node_inputs: HashMap::new(),
            aliases: HashMap::new(),
        }
    }
    fn visit(&mut self, graph: &'a GraphProto) {
        for node in &graph.node {
            let id = self.id_counter;
            self.id_counter += 1;
            self.node_map.insert(id, node);
            self.node_outputs.insert(id, node.output.clone());
            self.node_inputs.insert(id, node.input.clone());
            for input in &node.input {
                self.uses.entry(input.clone()).or_default().push(id);
            }
            // Recurse for subgraphs (If, Loop, Scan) in DETERMINISTIC ORDER matching generation
            if node.op_type == "If" {
                let then_g = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "then_branch")
                    .and_then(|a| a.g.as_ref());
                let else_g = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "else_branch")
                    .and_then(|a| a.g.as_ref());
                // Record aliases
                for (i, out_name) in node.output.iter().enumerate() {
                    let mut sources = Vec::new();
                    if let Some(g) = then_g
                        && i < g.output.len()
                    {
                        sources.push(g.output[i].name.clone());
                    }
                    if let Some(g) = else_g
                        && i < g.output.len()
                    {
                        sources.push(g.output[i].name.clone());
                    }
                    if !sources.is_empty() {
                        self.aliases.insert(out_name.clone(), sources);
                    }
                }
                if let Some(g) = then_g {
                    self.visit(g);
                }
                if let Some(g) = else_g {
                    self.visit(g);
                }
            } else if node.op_type == "Loop" {
                let body = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "body")
                    .and_then(|a| a.g.as_ref());
                // For Loop, output [i] corresponds to scan_output.
                // The main outputs are N iteration outputs + K scan outputs?
                // For simplicity, just aliasing implicit body outputs is harder because naming convention relies on scan_outputs.
                // Silero VAD mostly uses If, let's stick to If for now. Code generation for Loop is complex.
                if let Some(g) = body {
                    self.visit(g);
                }
            } else {
                // Record Zero-Copy Aliases
                if matches!(
                    node.op_type.as_str(),
                    "Squeeze" | "Unsqueeze" | "Reshape" | "Identity" | "Flatten" | "Cast"
                ) && !node.input.is_empty()
                    && !node.output.is_empty()
                {
                    self.aliases
                        .insert(node.output[0].clone(), vec![node.input[0].clone()]);
                }
                for attr in &node.attribute {
                    if let Some(g) = &attr.g {
                        self.visit(g);
                    }
                }
            }
        }
    }
}
pub(crate) struct AnalysisData {
    pub death_time: HashMap<String, usize>,
    pub uses: HashMap<String, Vec<usize>>,
}
fn solve_allocation(graph: &GraphProto, outputs: &[String]) -> (Allocator, AnalysisData) {
    let mut visitor = NodeVisitor::new();
    visitor.visit(graph);
    // Add graph outputs as "uses" at infinity (max_id)
    let max_id = visitor.id_counter;
    for out in outputs {
        visitor.uses.entry(out.clone()).or_default().push(max_id);
    }
    // Liveness Analysis
    let mut death_time = HashMap::new();
    for (name, uses) in &visitor.uses {
        if let Some(max) = uses.iter().max() {
            death_time.insert(name.clone(), *max);
        }
    }
    // Propagate aliases backwards
    // If output X aliases Y, then Y must live as long as X.
    let mut changed = true;
    while changed {
        changed = false;
        for (alias, sources) in &visitor.aliases {
            let alias_death = *death_time.get(alias).unwrap_or(&0);
            for source in sources {
                let source_death = *death_time.get(source).unwrap_or(&0);
                if alias_death > source_death {
                    death_time.insert(source.clone(), alias_death);
                    changed = true;
                }
            }
        }
    }
    let mut free_buffers: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    let mut active_allocations: HashMap<String, usize> = HashMap::new();
    let mut all_allocations: HashMap<String, usize> = HashMap::new();
    let mut max_buffers = 0;
    for id in 0..visitor.id_counter {
        // Release buffers
        let mut to_remove = Vec::new();
        for (tensor, &buf_idx) in &active_allocations {
            let die_at = *death_time.get(tensor).unwrap_or(&usize::MAX);
            if die_at < id {
                to_remove.push((tensor.clone(), buf_idx));
            }
        }
        for (t, b) in to_remove {
            active_allocations.remove(&t);
            free_buffers.push(Reverse(b));
        }
        // Allocate output buffers
        if let Some(node_outputs) = visitor.node_outputs.get(&id) {
            for out in node_outputs {
                if out.is_empty() {
                    continue;
                }
                // Check if this output aliases a source
                let mut alias_idx = None;
                if let Some(sources) = visitor.aliases.get(out) {
                    // Try to find the buffer of the source
                    // Since we propagated liveness, the source buffer should be active
                    // For single source aliases (Identity etc)
                    if let Some(src) = sources.first() {
                        if let Some(&idx) = active_allocations.get(src) {
                            alias_idx = Some(idx);
                        } else if let Some(&_idx) = all_allocations.get(src) {
                            // Source might be dead but consistent?
                            // But we need it to be active to reserve it.
                            // If it's not active, maybe it was freed?
                            // But we propagated lifetime, so it SHOULD be active.
                            // Unless it's a constant or weight which has no buffer?
                            // Weights used to be handled separately.
                            // If source is not in active/all allocations, it might be an input or weight.
                            // In that case, we can't alias into a workspace buffer number??
                            // Wait, weights don't use workspace.
                            // Input tensors don't use workspace logic here?
                            // If source is weight, Alias is view of weight.
                            // The generator will handle view of weight.
                            // We just need to ensure we don't allocate a NEW buffer for it.
                        }
                    }
                }
                if let Some(idx) = alias_idx {
                    // It's an alias to an existing buffer.
                    // Do NOT mark as active allocation (owner responsible for freeing)
                    all_allocations.insert(out.clone(), idx);
                } else {
                    // Allocate new
                    // Check inputs to avoid reusing their buffers for the same node (Borrow Checker)
                    let mut input_bufs = std::collections::HashSet::new();
                    if let Some(inps) = visitor.node_inputs.get(&id) {
                        input_bufs
                            .extend(inps.iter().filter_map(|n| all_allocations.get(n)).cloned());
                    }
                    // Check inputs of preceding nodes to avoid buffer conflicts in fused
                    // patterns (e.g. Conv+Sigmoid+Mul → Conv2d+SiLU consumes 3 nodes,
                    // Triple SiLU consumes 6 nodes)
                    for prev in 1..=5 {
                        if id >= prev
                            && let Some(inps) = visitor.node_inputs.get(&(id - prev))
                        {
                            input_bufs.extend(
                                inps.iter().filter_map(|n| all_allocations.get(n)).cloned(),
                            );
                        }
                    }
                    let mut deferred = Vec::new();
                    let mut selected_buf = None;
                    while let Some(Reverse(b)) = free_buffers.pop() {
                        if input_bufs.contains(&b) {
                            deferred.push(b);
                        } else {
                            selected_buf = Some(b);
                            break;
                        }
                    }
                    for b in deferred {
                        free_buffers.push(Reverse(b));
                    }
                    let buf_idx = if let Some(b) = selected_buf {
                        b
                    } else {
                        let b = max_buffers;
                        max_buffers += 1;
                        b
                    };
                    active_allocations.insert(out.clone(), buf_idx);
                    all_allocations.insert(out.clone(), buf_idx);
                }
            }
        }
    }
    (
        Allocator {
            tensor_to_buffer: all_allocations,
            num_buffers: max_buffers,
        },
        AnalysisData {
            death_time,
            uses: visitor.uses,
        },
    )
}
pub mod generate;
pub mod ir;
pub mod ops;
mod passes;

pub(crate) use generate::*;

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}

pub struct Compiler {
    pub(crate) overrides: HashMap<String, OpHandler>,
    pub(crate) model_name: String,
    pub(crate) patterns: Vec<Pattern>,
    pub(crate) custom_methods: Vec<String>,
    pub constant_folding: bool,
    pub(crate) fusion: bool,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            overrides: HashMap::new(),
            model_name: "Model".to_string(),
            patterns: Vec::new(),
            custom_methods: Vec::new(),
            constant_folding: false,
            fusion: false,
        }
    }
    /// Register a custom code generator for an operator type.
    /// The handler closure will be called instead of the default generation logic.
    pub fn with_override<F>(mut self, op: &str, handler: F) -> Self
    where
        F: Fn(&NodeProto, &[String], &[String], &str, &mut dyn Write, usize) -> std::io::Result<()>
            + 'static,
    {
        self.overrides.insert(op.to_string(), Box::new(handler));
        self
    }
    /// Register a pattern matcher and generator.
    pub fn with_pattern<M, G>(mut self, name: &str, matcher: M, generator: G) -> Self
    where
        M: Fn(&[&NodeProto]) -> Option<usize> + 'static,
        G: Fn(
                &[&NodeProto],
                &HashMap<String, (usize, usize, Vec<usize>, i32)>,
                Option<&Allocator>,
                &mut dyn Write,
                usize,
            ) -> std::io::Result<()>
            + 'static,
    {
        self.patterns.push(Pattern {
            name: name.to_string(),
            matcher: Box::new(matcher),
            generator: Box::new(generator),
        });
        self
    }
    /// Add a custom method to the implementation block.
    pub fn with_custom_method(mut self, code: &str) -> Self {
        self.custom_methods.push(code.to_string());
        self
    }

    /// Fuses operator chains (LayerNorm, GELU, SiLU, Conv + bias/activation,
    /// MatMul + Add, dynamically quantized linear layers, ...) into single
    /// operators with fused kernels.
    pub fn with_default_optimizations(mut self) -> Self {
        self.fusion = true;
        self
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.model_name = to_pascal_case(name);
        self
    }

    pub fn with_constant_folding(mut self, enabled: bool) -> Self {
        self.constant_folding = enabled;
        self
    }

    pub fn compile(
        &self,
        graph: &GraphProto,
    ) -> Result<CompilationResult, Box<dyn std::error::Error>> {
        let mut ir = ir::Graph::from_onnx(graph.clone());
        // Must precede folding, which would otherwise consume the int8 weights.
        passes::fuse_qdq_matmul(&mut ir);
        if self.constant_folding {
            passes::fold_constants(&mut ir);
        }
        // Runs after folding so that only the activation pairs are left.
        passes::fuse_qdq(&mut ir);
        if self.fusion {
            passes::fuse(&mut ir);
        }
        passes::infer_dtypes(&mut ir);
        let var_types = variable_types(&ir);
        let graph_storage = ir.into_onnx();
        let graph = &graph_storage;
        let mut weights_data = Vec::new();
        let mut offset_map = Vec::new();
        let mut current_offset = 0usize;
        let mut int64_map = HashMap::new();
        let mut content_hash_map = HashMap::new();
        // 1. Collect Weights
        collect_weights(
            graph,
            &mut weights_data,
            &mut offset_map,
            &mut current_offset,
            &mut int64_map,
            &mut content_hash_map,
        )?;
        // Prepare weight lookup map for body generation
        let mut weight_lookup = HashMap::new();
        for (name, offset, len, shape, data_type) in &offset_map {
            let safe_name = sanitize_name(name);
            weight_lookup
                .entry(safe_name)
                .or_insert((*offset, *len, shape.clone(), *data_type));
        }
        let nodes: Vec<&NodeProto> = graph.node.iter().collect();

        // Run Allocation/Liveness FIRST
        let output_names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();
        let (allocator, analysis) = solve_allocation(graph, &output_names);
        let mut chunk_functions = Vec::new(); // Will hold the code for fn chunk_0(...) { ... }
        let mut body_buffer = Vec::new(); // Will hold the code for forward() body
        let mut current_id = 0;
        generate_partitioned_graph(
            &nodes,
            &mut chunk_functions,
            &mut body_buffer,
            2,
            &weight_lookup,
            &int64_map,
            Some(&allocator),
            Some(&analysis),
            &mut current_id,
            self,
            &graph.input,
            &graph.output,
            &var_types,
        )?;
        let body_str =
            String::from_utf8(body_buffer).map_err(|e| format!("UTF-8 error in body: {}", e))?;
        // Detect used buffers
        // We need to scan both body AND usage in chunk functions
        let mut all_code = body_str.clone();
        for func in &chunk_functions {
            all_code.push_str(func);
        }
        let mut used_buffers = std::collections::HashSet::new();
        let mut search_idx = 0;
        while let Some(idx) = all_code[search_idx..].find("ws.buf_") {
            let start = search_idx + idx + 7; // "ws.buf_".len()
            let mut end = start;
            while end < all_code.len() {
                let c = all_code.as_bytes()[end] as char;
                if !c.is_ascii_digit() {
                    break;
                }
                end += 1;
            }
            if end > start
                && let Ok(buf_idx) = all_code[start..end].parse::<usize>()
            {
                used_buffers.insert(buf_idx);
            }
            search_idx = end;
        }
        // 2. Generate Rust Code
        let mut code = Vec::new();
        // Header
        writeln!(&mut code, "// Auto-generated by lele_compiler")?;
        writeln!(&mut code, "#![allow(non_snake_case)]")?;
        writeln!(&mut code, "#![allow(unused_variables)]")?;
        writeln!(&mut code, "#![allow(dead_code)]")?;
        writeln!(&mut code, "#![allow(unused_parens)]")?;
        writeln!(&mut code, "#![allow(unused_mut)]")?;
        writeln!(&mut code, "#![allow(unused_imports)]")?;
        writeln!(&mut code, "#![allow(clippy::too_many_arguments)]")?;
        writeln!(&mut code, "use lele::tensor::TensorView;")?;
        writeln!(&mut code, "use lele::kernels::*;")?;
        writeln!(&mut code, "\n")?;
        let struct_name = &self.model_name;
        // Generate Workspace Struct
        writeln!(&mut code, "#[derive(Default)]")?;
        writeln!(&mut code, "pub struct {}Workspace {{", struct_name)?;
        for i in 0..allocator.num_buffers {
            if used_buffers.contains(&i) {
                writeln!(&mut code, "    pub buf_{}: Vec<f32>,", i)?;
            }
        }
        writeln!(&mut code, "}}")?;
        writeln!(&mut code, "impl {}Workspace {{", struct_name)?;
        writeln!(&mut code, "    pub fn new() -> Self {{")?;
        writeln!(&mut code, "        Self::default()")?;
        writeln!(&mut code, "    }}")?;
        writeln!(&mut code, "}}")?;
        // Main struct: wraps the weights and derefs to them, so the weight
        // accessors and fused ops in lele::runtime are called as `self.*`.
        writeln!(&mut code, "\npub struct {}<'a> {{", struct_name)?;
        writeln!(&mut code, "    weights: lele::runtime::Weights<'a>,")?;
        writeln!(&mut code, "}}")?;
        writeln!(
            &mut code,
            "\nimpl<'a> std::ops::Deref for {}<'a> {{",
            struct_name
        )?;
        writeln!(&mut code, "    type Target = lele::runtime::Weights<'a>;")?;
        writeln!(&mut code, "    fn deref(&self) -> &Self::Target {{")?;
        writeln!(&mut code, "        &self.weights")?;
        writeln!(&mut code, "    }}")?;
        writeln!(&mut code, "}}")?;
        writeln!(&mut code, "\nimpl<'a> {}<'a> {{", struct_name)?;
        writeln!(&mut code, "    pub fn new(data: &'a [u8]) -> Self {{")?;
        writeln!(
            &mut code,
            "        Self {{ weights: lele::runtime::Weights::new(data) }}"
        )?;
        writeln!(&mut code, "    }}")?;
        for method in &self.custom_methods {
            writeln!(&mut code, "{}", method)?;
        }
        // Generated Chunk Functions
        for func in &chunk_functions {
            writeln!(&mut code, "{}", func)?;
        }

        // Inference Function
        writeln!(&mut code, "\n    // Inference Entry Point")?;
        let args: Vec<String> = graph
            .input
            .iter()
            .map(|i| {
                let name = sanitize_name(&i.name);
                let ty = var_types.get(&name).map(|s| s.as_str()).unwrap_or("f32");
                if ty == "f32" {
                    format!("{}: TensorView<'a>", name)
                } else {
                    format!("{}: TensorView<'a, {}>", name, ty)
                }
            })
            .collect();
        // Generate return type
        let ret_types: Vec<String> = graph
            .output
            .iter()
            .map(|o| {
                let name = sanitize_name(&o.name);
                let ty = var_types.get(&name).map(|s| s.as_str()).unwrap_or("f32");
                if ty == "f32" {
                    "TensorView<'a>".to_string()
                } else {
                    format!("TensorView<'a, {}>", ty)
                }
            })
            .collect();
        let ret_sig = if ret_types.len() == 1 {
            ret_types[0].clone()
        } else {
            format!("({})", ret_types.join(", "))
        };
        // forward (legacy)
        writeln!(
            &mut code,
            "    pub fn forward(&self, {}) -> {} {{",
            args.join(", "),
            ret_sig.replace("'a", "'static")
        )?;
        writeln!(
            &mut code,
            "        let mut ws = {}Workspace::new();",
            struct_name
        )?;
        let arg_names_str = graph
            .input
            .iter()
            .map(|i| sanitize_name(&i.name))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            &mut code,
            "        let res = self.forward_with_workspace(&mut ws, {});",
            arg_names_str
        )?;
        let ret_vals_owned: Vec<String> = if graph.output.len() == 1 {
            vec!["res.to_owned()".to_string()]
        } else {
            (0..graph.output.len())
                .map(|i| format!("res.{}.to_owned()", i))
                .collect()
        };
        let ret_str_owned = if ret_vals_owned.len() == 1 {
            ret_vals_owned[0].clone()
        } else {
            format!("({})", ret_vals_owned.join(", "))
        };
        writeln!(&mut code, "        {}", ret_str_owned)?;
        writeln!(&mut code, "    }}")?;

        // forward_with_workspace
        writeln!(
            &mut code,
            "    pub fn forward_with_workspace<'w>(&self, ws: &'w mut {}Workspace, {}) -> {} {{",
            struct_name,
            args.join(", ").replace("'a", "'w"),
            ret_sig.replace("'a", "'w")
        )?;
        // Generate Body
        code.extend_from_slice(body_str.as_bytes());
        // Return outputs
        let ret_vals: Vec<String> = graph
            .output
            .iter()
            .map(|o| {
                let name = sanitize_name(&o.name);
                let target_type = var_types.get(&name).map(|s| s.as_str()).unwrap_or("f32");
                if let Some((offset, len, shape, data_type)) = weight_lookup.get(&name) {
                    weight_expr(*offset, *len, shape, *data_type, target_type)
                } else {
                    name
                }
            })
            .collect();
        if ret_vals.len() == 1 {
            writeln!(&mut code, "        {}", ret_vals[0])?;
        } else {
            writeln!(&mut code, "        ({})", ret_vals.join(", "))?;
        }
        writeln!(&mut code, "    }}")?; // End forward
        writeln!(&mut code, "}}")?; // End impl
        Ok(CompilationResult {
            code: String::from_utf8(code).map_err(|e| format!("UTF-8 error: {}", e))?,
            weights: weights_data,
        })
    }
}
pub fn sanitize_name(name: &str) -> String {
    let s = name
        .replace(".", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_");
    if s.chars().next().is_some_and(|c| c.is_numeric()) {
        format!("_{}", s)
    } else {
        s
    }
}

fn compute_hash(bytes: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn collect_weights(
    graph: &GraphProto,
    bin_data: &mut Vec<u8>,
    offset_map: &mut Vec<(String, usize, usize, Vec<usize>, i32)>,
    current_offset: &mut usize,
    int64_map: &mut HashMap<String, (Vec<i64>, Vec<usize>)>,
    content_hash_map: &mut HashMap<u64, (usize, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initializers
    for init in &graph.initializer {
        // Populate int64_map for INT64 types
        if init.data_type == 7 || init.data_type == 6 {
            // INT64 or INT32
            if let Ok((data, dims)) = tensor_to_array(init) {
                let ints: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                int64_map.insert(init.name.clone(), (ints, dims));
            }
        }

        if let Ok((bytes, shape, data_type)) = crate::model::tensor_to_vec_u8(init)
            && !bytes.is_empty()
        {
            // Compute content hash for deduplication
            let hash = compute_hash(&bytes);

            // Check if we already have identical weight content
            if let Some(&(existing_offset, existing_len)) = content_hash_map.get(&hash) {
                // Reuse existing offset, but keep current weight's shape and dtype
                offset_map.push((
                    init.name.clone(),
                    existing_offset,
                    existing_len,
                    shape,
                    data_type,
                ));
            } else {
                // New weight: align, write, and record
                let remainder = *current_offset % 16;
                if remainder != 0 {
                    let padding = 16 - remainder;
                    bin_data.write_all(&vec![0u8; padding])?;
                    *current_offset += padding;
                }

                bin_data.write_all(&bytes)?;
                offset_map.push((
                    init.name.clone(),
                    *current_offset,
                    bytes.len(),
                    shape.clone(),
                    data_type,
                ));
                content_hash_map.insert(hash, (*current_offset, bytes.len()));
                *current_offset += bytes.len();
            }
        }
    }
    // 2. Constants
    for node in &graph.node {
        if node.op_type == "Constant"
            && let Some(attr) = node.attribute.iter().find(|a| a.name == "value")
            && let Some(t) = &attr.t
        {
            // Populate int64_map for INT64 types
            if t.data_type == 7 || t.data_type == 6 {
                // INT64 or INT32
                if let Ok((data, dims)) = tensor_to_array(t) {
                    let ints: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                    if let Some(out_name) = node.output.first() {
                        int64_map.insert(out_name.clone(), (ints, dims));
                    }
                }
            }

            if let Ok((bytes, shape, data_type)) = crate::model::tensor_to_vec_u8(t)
                && !bytes.is_empty()
            {
                // Compute content hash for deduplication
                let hash = compute_hash(&bytes);

                // Check if we already have identical weight content
                if let Some(&(existing_offset, existing_len)) = content_hash_map.get(&hash) {
                    // Reuse existing offset, but keep current weight's shape and dtype
                    if let Some(out_name) = node.output.first() {
                        offset_map.push((
                            out_name.clone(),
                            existing_offset,
                            existing_len,
                            shape,
                            data_type,
                        ));
                    }
                } else {
                    // New weight: align, write, and record
                    let remainder = *current_offset % 16;
                    if remainder != 0 {
                        let padding = 16 - remainder;
                        bin_data.write_all(&vec![0u8; padding])?;
                        *current_offset += padding;
                    }

                    bin_data.write_all(&bytes)?;
                    if let Some(out_name) = node.output.first() {
                        offset_map.push((
                            out_name.clone(),
                            *current_offset,
                            bytes.len(),
                            shape.clone(),
                            data_type,
                        ));
                        content_hash_map.insert(hash, (*current_offset, bytes.len()));
                        *current_offset += bytes.len();
                    }
                }
            }
        }
        // 3. Recurse
        for attr in &node.attribute {
            if let Some(g) = &attr.g {
                collect_weights(
                    g,
                    bin_data,
                    offset_map,
                    current_offset,
                    int64_map,
                    content_hash_map,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod conv_pattern_tests {
    use super::*;
    use crate::model::onnx_proto::{AttributeProto, TensorProto, ValueInfoProto};

    /// A depthwise 3x3 Conv with stride `[2, 1]` (as in PP-OCR's recognizer),
    /// followed by `next` (`Relu`, or `Add` of a bias initializer).
    fn graph(next: &str) -> GraphProto {
        let ints = |name: &str, v: &[i64]| AttributeProto {
            name: name.to_string(),
            ints: v.to_vec(),
            ..Default::default()
        };
        let mut group = ints("group", &[]);
        group.i = 4;
        let conv = NodeProto {
            op_type: "Conv".to_string(),
            input: vec!["x".to_string(), "w".to_string()],
            output: vec!["c".to_string()],
            attribute: vec![
                ints("kernel_shape", &[3, 3]),
                ints("strides", &[2, 1]),
                ints("dilations", &[1, 2]),
                ints("pads", &[1, 2, 1, 2]),
                group,
            ],
            ..Default::default()
        };
        let mut inputs = vec!["c".to_string()];
        if next == "Add" {
            inputs.push("b".to_string());
        }
        let after = NodeProto {
            op_type: next.to_string(),
            input: inputs,
            output: vec!["y".to_string()],
            ..Default::default()
        };
        let init = |name: &str, dims: &[i64]| TensorProto {
            name: name.to_string(),
            data_type: 1,
            dims: dims.to_vec(),
            raw_data: vec![0u8; 4 * dims.iter().product::<i64>() as usize],
            ..Default::default()
        };
        GraphProto {
            node: vec![conv, after],
            initializer: vec![init("w", &[4, 1, 3, 3]), init("b", &[1, 4, 1, 1])],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// The fused Conv patterns used to keep only the first stride and
    /// dilation and repeat it, turning `[2, 1]` into `[2, 2]`.
    #[test]
    fn fused_conv_keeps_per_axis_strides_and_dilations() {
        for next in ["Relu", "Add"] {
            let result = Compiler::new()
                .with_name("ConvStrides")
                .with_default_optimizations()
                .compile(&graph(next))
                .expect("compilation should succeed");
            let call = result
                .code
                .lines()
                .find(|l| l.contains("lele::kernels::conv2d"))
                .unwrap_or_else(|| panic!("{next}: no conv2d call in\n{}", result.code));
            assert!(
                call.contains("&[1, 2], 4, &[1, 2, 1, 2], &[2, 1]"),
                "{next}: {call}"
            );
            // Both went through their fused pattern.
            let fused = if next == "Relu" {
                "conv2d_fused("
            } else {
                "Some(&self.weight_f32"
            };
            assert!(call.contains(fused), "{next}: {call}");
        }
    }
}
