use std::fs::File;
use std::vec;
//use std::io::Write;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, masked_softmax, matmul_transb, random_sample, rms_norm, swiglu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
//use serde::ser::SerializeTupleStruct;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        // let file_result = File::create("tensor0.txt");
        // let mut file = match file_result {
        //                             Ok(file) => file,
        //                             Err(e) => {
        //                                 eprintln!("无法创建文件 'output.txt': {}", e);
        //                                 return residual;
        //                             }
        // };
        // if let Err(e) = writeln!(file, "{:?}",residual.data()) {
        //         eprintln!("写入文件失败: {}", e);
        //         return residual;
        // }

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states, // (seq, n_kv_h * n_groups * dqkv)
                &mut att_scores,    // (n_kv_h, n_groups, seq, total_seq)
                q,                 
                full_k,                
                full_v,                 
                self.n_kv_h, n_groups, seq_len, total_seq_len,self.dqkv,);
            // // println!("hidden_states_{layer}");
            // if let Err(e) = writeln!(file, "hidden_states_{}", layer) {
            //     eprintln!("写入文件失败: {}", e);
            //     return hidden_states;
            // }
            // // hidden_states.print();
            // if let Err(e) = writeln!(file, "{:?}",hidden_states.data()) {
            //     eprintln!("写入文件失败: {}", e);
            //     return hidden_states;
            // }
          
            // out = attn_V @ O_weight.T
            // residual = out + residual
            OP::matmul_transb(&mut residual, 1.0, &hidden_states, &self.params.wo[layer], 1.0);

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            ) 
        }
        //打印张量
        // let file_result = File::create("tensor.txt");
        // let mut file = match file_result {
        //     Ok(file) => file,
        //     Err(e) => {
        //         eprintln!("无法创建文件 'output.txt': {}", e);
        //         return hidden_states;
        //     }
        // };           
        // if let Err(e) = writeln!(file, "{:?}",hidden_states.data()) {
        //     eprintln!("写入文件失败: {}", e);
        //     return hidden_states;
        //     }
        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();       
        let mut kvcache =self.new_cache();

        // let input = Tensor::new(vec![self.bos_token_id], &vec![1]);
        // let logits = self.forward(&input, &mut kvcache);
        // let mut token_id = random_sample(&logits, top_p, top_k, temperature);
        // // let mut a = vec![];
        // // a.extend_from_slice(token_ids);
        // for i in 0..token_ids.len() {
        //     let input = Tensor::<u32>::new(vec![token_ids[i]], &vec![1]);
        //     let logits = self.forward(&input, &mut kvcache);
        //     token_id = OP::random_sample(&logits, top_p, top_k, temperature);
        // }        
        result.push(self.bos_token_id);
        result.extend_from_slice(token_ids);

        let input = Tensor::new(result.clone(), &vec![result.len()]);
        let logits = self.forward(&input, &mut kvcache);

        let mut token_id = random_sample(&logits, top_p, top_k, temperature);

        while result.len() <= max_len {
            result.push(token_id);
            if token_id == self.eos_token_id{
                break;}        
            let input =Tensor::new(vec![token_id], &vec![1]);
            let next_logits = self.forward(&input, &mut kvcache);
            token_id = random_sample(&next_logits, top_p, top_k, temperature);  
        }  
        result
    }

    pub fn chat_generate(
        &self,
        token_ids: &[u32],
        cache: &mut KVCache<f32>,
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();       
        //let mut kvcache =self.new_cache();
        //println!("{:?}",token_ids);
        let input = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        let logits = self.forward(&input, cache);
        let mut next_token = random_sample(&logits, top_p, top_k, temperature);

        result.push(next_token);

        while result.len() < max_len {
        
            if next_token == self.eos_token_id{
                break;}        
            let input =Tensor::new(vec![next_token], &vec![1]);
            let logits = self.forward(&input, cache);
            next_token = random_sample(&logits, top_p, top_k, temperature);  

            result.push(next_token);
        }  
        //println!("{:?}",result);
        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    //todo!("Implement self_attention");
    assert!(q.shape().len()==3);
    let n_q_heads = n_kv_h * n_groups;
    let mut att_scores:Vec<f32> = att_scores.data().to_vec();
    let mut attnv = vec![];
   
    for head in 0..n_q_heads{
        let mut q_r = vec![];
        let mut k_r =vec![];
        let mut v_r =vec![];
        for tok in 0..seq_len{
            for dim in 0..dqkv{
                let index = tok * (n_q_heads * dqkv) + head * dqkv + dim;                
                q_r.push(q.data()[index]);                
            }
        }        
        for tok in 0..total_seq_len {
            for dim in 0..dqkv {
                let index = tok * (n_kv_h * dqkv) + (head / n_groups) * dqkv + dim;//连续4个Q乘一个KV
                k_r.push(k.data()[index]);  
                v_r.push(v.data()[index]);             
            }           
        }
        let mut _v =vec![0.;v_r.len()];
        for dim in 0..dqkv{
            for tok in 0..total_seq_len{_v[dim*total_seq_len+tok] = v_r[tok*dqkv+dim];}
        }
        let v_head = Tensor::<f32>::new(_v, &vec![dqkv,total_seq_len]);
        let q_head = Tensor::<f32>::new(q_r, &vec![seq_len, dqkv]);
        let k_head = Tensor::<f32>::new(k_r, &vec![total_seq_len, dqkv]);
        let mut score_head = Tensor::<f32>::default(&vec![seq_len, total_seq_len]); 
        let mut attnv_head = Tensor::<f32>::default(&vec![seq_len, dqkv]);
        // println!("q_head_{}=",head);
        // q_head.print();
        // println!("k_head_{}=",head); 
        // k_head.print();

        //score = Q @ K.T / sqrt(dim)
        OP::matmul_transb(&mut score_head, 0., &q_head, &k_head, 1.0/(dqkv as f32).sqrt());       
        //attn = softmax(score)
        masked_softmax(&mut score_head);
        score_head.data().iter().for_each(|&i|att_scores.push(i));

        OP::matmul_transb(&mut attnv_head, 0., &score_head, &v_head, 1.0);//attn_V = attn @ V
        attnv_head.data().iter().for_each(|&i|attnv.push(i));
    }
    //att_scores.print();
    //reshape hidden_states [seq_len, n_kv_h*n_groups, dqkv]
    let hidden = unsafe {hidden_states.data_mut()};
    for i in 0..n_q_heads{
        for j in 0..seq_len {
            for d in 0..dqkv {
                hidden[j*n_q_heads*dqkv+i*dqkv+d] = attnv[i*seq_len*dqkv+j*dqkv+d];}
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    //todo!("Implement mlp");
    rms_norm(hidden_states, residual, rms_w, eps);//hidden = rms_norm(residual)
    //println!("{:?}",hidden_states.data());
    matmul_transb(gate, 0., hidden_states, w_gate, 1.);//gate = hidden @ gate_weight.T
    //println!("{:?}",gate.data());
    matmul_transb(up, 0., hidden_states, w_up, 1.);//up = hidden @ up_weight.T
    swiglu(up, gate);//act = gate * sigmoid(gate) * up
    matmul_transb(residual, 1., up, w_down, 1.);//output = act @ down_weight.T

}


#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 32002);
    assert_eq!(model.n_layers, 10);
    assert_eq!(model.n_q_h, 12);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 312);
    assert_eq!(model.dqkv, 26);
    assert_eq!(model.di, 1092);

    // assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    // assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    // assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    // assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    // assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    // assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    // assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    // assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    // assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    // assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    // assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    // assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
