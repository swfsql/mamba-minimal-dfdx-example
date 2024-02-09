//! This is a copy or adaption from https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/.

#![allow(clippy::erasing_op)]

pub mod mamba;

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use dfdx::prelude::*;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const MAMBA_MODEL_ID: &str = "state-spaces/mamba-130m";
const MAMBA_REVISION: &str = "refs/pr/1";
const MAMBA_CONFIG: &str = "config.json";
const MAMBA_FILENAMES: &str = "model.safetensors";
const TOKENIZER_MODEL_ID: &str = "EleutherAI/gpt-neox-20b";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

struct TextGeneration {
    model: mamba::Mamba<f32, Cpu>,
    device: candle_core::Device,
    cpu: Cpu,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

fn main() -> anyhow::Result<()> {
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let tokenizer_filename = api
        .model(TOKENIZER_MODEL_ID.into())
        .get(TOKENIZER_FILENAME)?;
    println!("tokenizer {TOKENIZER_FILENAME} path: {tokenizer_filename:?}");

    let repo = api.repo(Repo::with_revision(
        MAMBA_MODEL_ID.into(),
        RepoType::Model,
        MAMBA_REVISION.into(),
    ));
    // let mamba_config_filename = repo.get(MAMBA_CONFIG)?;
    // println!("mamba {MAMBA_CONFIG} path: {mamba_config_filename:?}");
    let mamba_filenames = vec![repo.get(MAMBA_FILENAMES)?];
    println!("mamba {MAMBA_FILENAMES} path: {mamba_filenames:?}");
    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let cpu = Cpu::default();

    let start = std::time::Instant::now();
    println!("started loading the model");
    let m = {
        let n_layer = 24;
        let padded_vocab_size = 50280;
        let d_model = 768;
        let m =
            mamba::MambaConfig::new(n_layer, padded_vocab_size, d_model, None, None, None, None);
        let mut m: mamba::Mamba<f32, Cpu> = cpu.try_build_module::<f32>(m)?;
        let load_renames = mamba::load::load_renames(n_layer);
        let skip_missing = true;
        let mut key_map = |key: String| load_renames.get(&key).unwrap_or(&key).to_string();
        m.load_safetensors_with(&mamba_filenames[0], skip_missing, &mut key_map)?;
        m
    };
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        m,
        tokenizer,
        299792458,
        None,
        None,
        1.1,
        1024, // 64
        &candle_examples::device(true)?,
        cpu.clone(),
    );

    let stateful = true;
    let stop_on_eos = false;
    pipeline.run("Mamba is the", 5000, stateful, stop_on_eos)?;
    // pipeline.run("Mamba is the", 30, !stateful, stop_on_eos)?;

    Ok(())
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
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

    fn run(
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

    fn add_logits(
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
