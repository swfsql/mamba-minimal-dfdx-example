//! This is a copy or adaption from https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/.

#![cfg_attr(target_arch = "wasm32", feature(slice_pattern))]
#![allow(clippy::erasing_op)]

pub mod mamba;
pub mod token_output_stream;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use candle_transformers::generation::LogitsProcessor;
use dfdx::prelude::*;
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;

// // TODO: lib.rs doesn't use hf_hub, only main does
// #[cfg(not(target_arch = "wasm32"))]
// use hf_hub::{api::sync::Api, Repo, RepoType};

pub mod hf_mamba {
    pub const MAMBA_MODEL_ID: &str = "state-spaces/mamba-130m";
    pub const MAMBA_REVISION: &str = "refs/pr/1";
    pub const MAMBA_CONFIG: &str = "config.json";
    pub const MAMBA_FILENAMES: &str = "model.safetensors";
    pub const TOKENIZER_MODEL_ID: &str = "EleutherAI/gpt-neox-20b";
    pub const TOKENIZER_FILENAME: &str = "tokenizer.json";
}
use hf_mamba::*;

pub struct TextGeneration {
    model: mamba::Mamba<f32, Cpu>,
    device: candle_core::Device,
    cpu: Cpu,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: mamba::Mamba<f32, Cpu>,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &candle_core::Device,
        cpu: Cpu,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            cpu,
        }
    }

    pub fn run(
        &mut self,
        prompt: &str,
        sample_len: usize,
        stateful: bool,
        stop_on_eos: bool,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        // only used for stateful
        let mut states = vec![];
        if stateful {
            for _ in 0..self.model.layers.len() {
                let state =
                    self.cpu
                        .try_build_module::<f32>(dfdx::nn::MambaStateCacheConfig::new(
                            1,
                            16,
                            4,
                            768 * 2,
                        ))?;
                states.push(state);
            }
        }

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let start_gen = std::time::Instant::now();
        let mut i = 0;
        while i < sample_len {
            // stateful
            if stateful {
                let input = self
                    .cpu
                    .tensor_from_vec(vec![tokens[i]], (1,))
                    .to_dtype::<usize>();
                let input = (input, states);
                let (logits, new_states) = <mamba::Mamba<f32, Cpu> as Module<
                    mamba::stateful::VocabInputWithStates<f32, Cpu, NoneTape>,
                >>::try_forward(&self.model, input)?;
                states = new_states;
                let shape = (logits.shape().1,);
                let logits = candle_core::Tensor::from_vec(logits.as_vec(), shape, &self.device)?
                    .to_dtype(candle_core::DType::F32)?;

                let next_token = self.add_logits(i, &mut tokens, &mut generated_tokens, logits)?;

                if next_token == eos_token {
                    println!();
                    println!("<|endoftext|>");
                    if stop_on_eos {
                        return Ok(());
                    }
                }
                i += 1;
            }
            // stateless
            else {
                let input = self
                    .cpu
                    .tensor_from_vec(tokens.clone(), (1, tokens.len()))
                    .to_dtype::<usize>();
                let logits_list = <mamba::Mamba<f32, Cpu> as Module<
                    mamba::stateless::VocabInput<Cpu, NoneTape>,
                >>::try_forward(&self.model, input)?;
                let shape = (logits_list.shape().2,);
                let logits_list = logits_list.as_vec();

                // logits contains an output for each timestep
                let logits_list = logits_list
                    .chunks_exact(50280)
                    .map(|chunk| candle_core::Tensor::from_slice(chunk, shape, &self.device))
                    .skip(i)
                    .collect::<Result<Vec<_>, _>>()?;
                for logits in logits_list.into_iter() {
                    let next_token =
                        self.add_logits(i, &mut tokens, &mut generated_tokens, logits)?;
                    if next_token == eos_token {
                        println!();
                        println!("<|endoftext|>");
                    }
                    if stop_on_eos {
                        return Ok(());
                    }
                    i += 1;
                }
            };
        }

        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }

    pub fn add_logits(
        &mut self,
        i: usize,
        tokens: &mut Vec<u32>,
        generated_tokens: &mut usize,
        logits: candle_core::Tensor,
    ) -> anyhow::Result<u32> {
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = i.saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &tokens[start_at..i + 1],
            )?
        };

        let next_token;
        if i + 1 < tokens.len() {
            // don't try to predict the next token (it was pre-defined)
            // also don't increment the "tokens" list (this token was already part of the list)
            next_token = tokens[i + 1];

            // should it still sample? idk
            // let _discarded_token = self.logits_processor.sample(&logits)?;
        } else {
            // try to predict the next token
            next_token = self.logits_processor.sample(&logits)?;
            // add the token to the "tokens" list
            tokens.push(next_token);
            *generated_tokens += 1;
        }

        // if the token has some valid representation, print it
        if let Some(t) = self.tokenizer.next_token(next_token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
        Ok(next_token)
    }
}
