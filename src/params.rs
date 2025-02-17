use std::vec;

use crate::config::LlamaConfigJson;
use crate::tensor::{self, Tensor};
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        
        let get_tensor = |name: &str|{
            let tensor_view = safetensor.tensor(name).unwrap();
            let tensor_data: Vec<f32> = tensor_view.data()
                                                   .chunks_exact(4)
                                                   .filter_map(|chunk|{
                                                    chunk.try_into().ok().map(|array:[u8;4]| f32::from_le_bytes(array))
                                                   })
                                                   .collect();
            Tensor::<f32>::new(tensor_data,&tensor_view.shape().to_vec())
        };

        //let n_layers = config.num_hidden_layers;
        
        //println!("{}",n_layers);
    

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: vec![get_tensor("model.layers.0.input_layernorm.weight"),get_tensor("model.layers.1.input_layernorm.weight")],
            wq: vec![get_tensor("model.layers.0.self_attn.q_proj.weight"),get_tensor("model.layers.1.self_attn.q_proj.weight")],
            wo: vec![get_tensor("model.layers.0.self_attn.o_proj.weight"),get_tensor("model.layers.1.self_attn.o_proj.weight")],
            wk: vec![get_tensor("model.layers.0.self_attn.k_proj.weight"),get_tensor("model.layers.1.self_attn.k_proj.weight")],
            wv: vec![get_tensor("model.layers.0.self_attn.v_proj.weight"),get_tensor("model.layers.1.self_attn.v_proj.weight")],
            rms_ffn_w: vec![get_tensor("model.layers.0.post_attention_layernorm.weight"),get_tensor("model.layers.1.post_attention_layernorm.weight")],
            w_gate: vec![get_tensor("model.layers.0.mlp.gate_proj.weight"),get_tensor("model.layers.1.mlp.gate_proj.weight")],
            w_up: vec![get_tensor("model.layers.0.mlp.up_proj.weight"),get_tensor("model.layers.1.mlp.up_proj.weight")],
            w_down: vec![get_tensor("model.layers.0.mlp.down_proj.weight"),get_tensor("model.layers.1.mlp.down_proj.weight")],
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}

#[cfg(test)]
mod test{
    use super::*;
    use std::fs::File;    
    use std::path::{Path, PathBuf};
    
    //use crate::slice::IndexOp;
    //use proptest::prelude::*;
#[test]
fn load_param(){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let config = File::open(model_dir.as_path().join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
    let model_file = std::fs::read(model_dir.as_path().join("model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&model_file).unwrap();
    let params = LLamaParams::from_safetensors(&tensors, &config);
    
    for (name, tensor) in tensors.tensors() {
        println!("Tensor name: {}", name);
        println!("Tensor shape: {:?}", tensor.shape());
        //println!("Tensor data: {:?}", tensor.data());
    }
}
}



