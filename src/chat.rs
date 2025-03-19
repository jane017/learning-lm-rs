use std::io::{self, Write};
//use crate::kvcache::KVCache;
use crate::{kvcache::KVCache, model::Llama};
use tokenizers::Tokenizer;
//use serde::ser::SerializeTupleStruct;
use std::path::PathBuf;

struct Message{
    pub role: String,
    pub content: String,
}

impl Message{
    pub fn format(&self)-> String {
        format!("<|im_start|>{}\n{}<|im_end|>\n", self.role, self.content)
    }
}

pub struct Chat {
    messages:Vec<Message>,
    llama:Llama<f32>,
    tokenizer:Tokenizer,
    cache: KVCache<f32>,
}

impl Chat{
    pub fn new(model_dir:PathBuf)->Self {
        let llama = Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let messages = Vec::new();
        let cache = llama.new_cache();
        Self { messages, llama , tokenizer, cache }
    }

    pub fn add_message(&mut self, role: &str, content:&str) {
        self.messages.push(Message { role: role.to_string(), content: content.to_string()});
    }

    pub fn input_prompt(&mut self)->String {
        let mut prompt = String::new();
        for msg in &self.messages{
            prompt.push_str(&msg.format());
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    pub fn chat(&mut self) {

        //let mut cache = self.llama.new_cache();

        self.add_message("system", "You are a helpful assistant");

        loop {
            print!("=========> ");
            io::stdout().flush().unwrap();
            let mut user_input =String::new();
            io::stdin().read_line(&mut user_input).unwrap();
            let user_input = user_input.trim();

            self.add_message("user", user_input);
            let input = self.input_prompt();

            let encoding = self.tokenizer.encode(input, true).unwrap();
            let input_ids =encoding.get_ids();

            let output_ids = self.llama.chat_generate(input_ids,&mut self.cache,500,0.8,30,1.,);
            let response = self.tokenizer.decode(&output_ids, true).unwrap();

            println!("Assistant: {}", response);
            self.add_message("assistant", &response);
        }
        

    }
}

#[test]

fn test() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    println!("{:?}",model_dir);
    let mut chat_ = Chat::new(model_dir);
    chat_.chat();
}
