mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod chat;

use std::io; 
use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::chat::Chat;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    println!("Choose the number of function: 1. A story ; 2. A chat");

    let mut user_input =String::new();
    io::stdin().read_line(&mut user_input).unwrap();
    let user_input = user_input.trim();

    if user_input == "1" {
        let model_dir = PathBuf::from(project_dir).join("models").join("story");
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let input = "Once upon a time";
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        //print!("\n{}", input);
        let output_ids = llama.generate(input_ids,150,0.8,30,1.,);
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }else if user_input == "2"{
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        let mut chat_ = Chat::new(model_dir);
        chat_.chat();

    }else {
        print!("NO this function , please check your input!");
    }
    
}
