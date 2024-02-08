//! This is a copy or adaption from https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/.

#![allow(clippy::erasing_op)]

pub mod mamba;

use dfdx::prelude::*;

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::tensor::SafeTensors;
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

    

    fn run_stateless(&mut self, prompt: &str, sample_len: usize) -> anyhow::Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t)? {
                print!("{t}")
            }
        }

        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let start_gen = std::time::Instant::now();
        let mut i = 0;
        'sample: while i < sample_len {
            // dbg!(i);
            let input = self
                .cpu
                .tensor_from_vec(tokens.clone(), (1, tokens.len()))
                .to_dtype::<usize>();
            let logits_list = <mamba::Mamba<f32, Cpu> as Module<
                mamba::stateless::VocabInput<Cpu, NoneTape>,
            >>::try_forward(&self.model, input)?;
            // let logits = self.model.try_forward(input)?;
            let shape = (1, 1, logits_list.shape().2);
            let logits_list = logits_list.as_vec();

            // logits contains an output for each timestep
            let logits_list = logits_list
                .chunks_exact(50280)
                .map(|chunk| {
                    candle_core::Tensor::from_slice(chunk, shape, &self.device)
                        .and_then(|logits| logits.squeeze(0))
                        .and_then(|logits| logits.squeeze(0))
                        .and_then(|logits| logits.to_dtype(candle_core::DType::F32))
                })
                .collect::<Result<Vec<_>, _>>()?;

            // skip the logit outputs in which were already viewed
            // note: except for the first iteration, this will always only get the last logit
            // dbg!(logits_list.len());
            for logits in logits_list.into_iter().skip(i) {
                // dbg!(i);
                // panic!("{:?}", &logits.to_vec1::<f32>().unwrap()[0..20]);
                // println!("{:?}", &logits.to_vec1::<f32>().unwrap()[0..20]);

                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    // logits
                    // TODO: maybe i+1
                    let start_at = i.saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &tokens[start_at..i + 1],
                    )?
                };

                let mut next_token;
                // next_token = self.logits_processor.sample(&logits)?;
                if i + 1 < tokens.len() {
                    // dbg!("predefined");
                    // don't try to predict the next token (it was pre-defined)
                    next_token = tokens[i + 1];
                    // also don't increment the "tokens" list (this token was already part of the list)
                } else {
                    // dbg!("new");
                    // try to predict the next token
                    next_token = self.logits_processor.sample(&logits)?;
                    // add the token to the "tokens" list
                    tokens.push(next_token);
                    generated_tokens += 1;
                }
                if next_token == eos_token {
                    println!();
                    println!("Mamba signals the end of the output. But continuing anyway, with the same state...")
                    // break 'sample;
                }

                // if the token has some valid representation, print it
                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    // println!("{t}({i})");
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                i += 1;
            }
            // dbg!(i);
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

    fn run_stateful(&mut self, prompt: &str, sample_len: usize) -> anyhow::Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t)? {
                print!("{t}")
            }
        }

        std::io::stdout().flush()?;

        // statefull
        let mut states = vec![];
        for _ in 0..self.model.layers.len() {
            let state = self
                .cpu
                .try_build_module::<f32>(dfdx::nn::MambaStateCacheConfig::new(1, 16, 4, 768 * 2))?;
            // println!("{:?}", &state.conv_state.as_vec()[0..10]);
            states.push(state);
        }

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let start_gen = std::time::Instant::now();
        for i in 0..sample_len {
            // statefull
            // print!("input token: {{{}}}=", tokens[i]);
            let input = self
                .cpu
                .tensor_from_vec(vec![tokens[i]], (1,))
                .to_dtype::<usize>();
            let input = (input, states);
            let (logits, new_states) = <mamba::Mamba<f32, Cpu> as Module<
                mamba::stateful::VocabInputWithStates<f32, Cpu, NoneTape>,
            >>::try_forward(&self.model, input)?;
            // let (logits, new_states) = self.model.try_forward(input)?;
            states = new_states;
            // println!("{:?}", &logits.as_vec()[0..10]);
            let shape = (1, 1, logits.shape().1);
            let logits = candle_core::Tensor::from_vec(logits.as_vec(), shape, &self.device)?;

            let logits = logits
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(candle_core::DType::F32)?;

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let mut next_token = self.logits_processor.sample(&logits)?;
            // print!("output token: {{{}}}=", next_token);

            if i + 1 < tokens.len() {
                next_token = tokens[i + 1];
            } else {
                tokens.push(next_token);
                generated_tokens += 1;
                if next_token == eos_token {
                    println!();
                    println!("Mamba signals the end of the output. But continuing anyway, with the same state...")
                    // break;
                }
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                // print!("[{t}]");
                std::io::stdout().flush()?;
            }
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
}

fn run() -> anyhow::Result<()> {
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

    // let mamba_config = std::fs::read(mamba_config_filename)?;
    // let config = String::from_utf8_lossy(&mamba_config);
    // dbg!(&config);

    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let cpu = Cpu::default();

    use memmap2::MmapOptions;
    use std::fs::File;

    // let file = File::open(&mamba_filenames[0])?;
    // let buffer = unsafe { MmapOptions::new().map(&file)? };
    // let buffer = unsafe { MmapOptions::new().map(&file)? };
    // let tensors = SafeTensors::deserialize(&buffer)?;
    // for (name, view) in tensors.tensors() {
    //     let dtype = view.dtype();
    //     let shape = view.shape();
    //     println!("{name}: {dtype:?} - {shape:?}");
    // }

    // const MAMBA_TEST_PATH: &str =
    //     "/workspaces/coursera-deep-learning-specialization/r/src/c5/w1/extra/s4/mamba.safetensor";
    // let m = mamba_conv::MambaOptions::with(1, 16, 8).mamba_config();
    // let mut m = cpu.try_build_module::<f32>(m)?;
    // m.save_safetensors(MAMBA_TEST_PATH)?;
    // use memmap2::MmapOptions;
    // use std::fs::File;
    // let file = File::open(MAMBA_TEST_PATH)?;
    // let buffer = unsafe { MmapOptions::new().map(&file)? };
    // let tensors = SafeTensors::deserialize(&buffer)?;
    // for (name, view) in tensors.tensors() {
    //     let dtype = view.dtype();
    //     let shape = view.shape();
    //     println!("{name}: {dtype:?} - {shape:?}");
    // }

    let start = std::time::Instant::now();

    let m = mamba::MambaConfig::new(24, 50280, 768, None, None, None, None);
    let mut m: mamba::Mamba<f32, Cpu> = cpu.try_build_module::<f32>(m)?;
    // m.load_safetensors(&mamba_filenames[0])?;
    //
    let file = File::open(&mamba_filenames[0])?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let st = SafeTensors::deserialize(&buffer)?;
    m.embedding
        .weight
        .load_safetensor(&st, "backbone.embedding.weight")?;
    for (i, layer) in m.layers.iter_mut().enumerate() {
        println!("Loading mamba block layer {i}/23");
        let li = format!("backbone.layers.{i}");
        // mixer (mamba block)
        layer
            .res
            .0
             .1
            .in_proj
            .weight
            .load_safetensor(&st, &format!("{li}.mixer.in_proj.weight"))?;
        layer
            .res
            .0
             .1
            .conv1d
            .weight
            .load_safetensor(&st, &format!("{li}.mixer.conv1d.weight"))?;
        layer
            .res
            .0
             .1
            .conv1d_bias
            .bias
            .load_safetensor(&st, &format!("{li}.mixer.conv1d.bias"))?;
        layer
            .res
            .0
             .1
            .x_proj
            .weight
            .load_safetensor(&st, &format!("{li}.mixer.x_proj.weight"))?;
        layer
            .res
            .0
             .1
            .dt_proj
            .weight
            .load_safetensor(&st, &format!("{li}.mixer.dt_proj.weight"))?;
        layer
            .res
            .0
             .1
            .dt_proj
            .bias
            .load_safetensor(&st, &format!("{li}.mixer.dt_proj.bias"))?;
        layer
            .res
            .0
             .1
            .a_log
            .load_safetensor(&st, &format!("{li}.mixer.A_log"))?;
        layer
            .res
            .0
             .1
            .d
            .load_safetensor(&st, &format!("{li}.mixer.D"))?;
        layer
            .res
            .0
             .1
            .out_proj
            .weight
            .load_safetensor(&st, &format!("{li}.mixer.out_proj.weight"))?;
        // norm
        layer
            .res
            .0
             .0
            .gamma
            .load_safetensor(&st, &format!("{li}.norm.weight"))?;
    }
    m.norm_f
        .gamma
        .load_safetensor(&st, "backbone.norm_f.weight")?;
    m.lm_head.weight = m.embedding.weight.clone();

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        m,
        tokenizer,
        299792458,
        None,
        None,
        1.1,
        64,
        &candle_examples::device(false)?,
        cpu.clone(),
    );
    // pipeline.run_stateless("Mamba is the", 100)?; // works ok
    pipeline.run_stateful("Mamba is the", 5000)?;

    // // let prompt = "A";
    // let prompt = "Mamba is the";
    // let sample_len = 5000;

    // let mut tokens = tokenizer
    //     .encode(prompt, true)
    //     .map_err(anyhow::Error::msg)?
    //     .get_ids()
    //     .to_vec();

    // println!("input tokens:");
    // for &t in tokens.iter() {
    //     if let Some(ts) = tokenizer.id_to_token(t) {
    //         print!("{ts}");
    //         // println!("{ts} ({t})");
    //     }
    // }
    // // print!("");

    // let mut generated_tokens = 0usize;
    // let eos_token = match tokenizer.token_to_id("<|endoftext|>") {
    //     Some(token) => token,
    //     None => anyhow::bail!("cannot find the </s> token"),
    // };

    // let start_gen = std::time::Instant::now();
    // for _ in 0..sample_len {
    //     let input = cpu
    //         .tensor_from_vec(tokens.clone(), (1, tokens.len()))
    //         .to_dtype::<usize>();
    //     let logits = m.try_forward(input)?;
    //     // println!("{:?}", logits.shape());
    //     // let logits = logits.try_reshape_like(&(tokens.len(), 768));

    //     // println!("Gerou um logit..! {:?}", logits.shape());
    //     // panic!();
    //     // sample argmax
    //     let next_token = {
    //         let logits = logits.as_vec();
    //         logits
    //             .iter()
    //             .skip(50280 * (tokens.len() - 1))
    //             .enumerate()
    //             .max_by(|(_, u), (_, v)| u.total_cmp(v))
    //             .map(|(i, _)| i)
    //             .unwrap()
    //     };
    //     tokens.push(next_token as u32);
    //     generated_tokens += 1;
    //     if next_token as u32 == eos_token {
    //         break;
    //     }

    //     if let Some(t) = tokenizer.id_to_token(next_token as u32) {
    //         print!("{t}");
    //     } else {
    //         print!("_ ({})", next_token);
    //     }
    //     use std::io::Write;
    //     std::io::stdout().flush()?;

    //     if generated_tokens == 100 {
    //         panic!()
    //     }

    //     // let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
    //     // let logits = self.model.forward(&input)?;
    //     // let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    //     // let logits = if self.repeat_penalty == 1. {
    //     //     logits
    //     // } else {
    //     //     let start_at = tokens.len().saturating_sub(self.repeat_last_n);
    //     //     candle_transformers::utils::apply_repeat_penalty(
    //     //         &logits,
    //     //         self.repeat_penalty,
    //     //         &tokens[start_at..],
    //     //     )?
    //     // };

    //     // let next_token = self.logits_processor.sample(&logits)?;
    //     // tokens.push(next_token);
    //     // generated_tokens += 1;
    //     // if next_token == eos_token {
    //     //     break;
    //     // }
    //     // if let Some(t) = self.tokenizer.next_token(next_token)? {
    //     //     print!("{t}");
    //     //     std::io::stdout().flush()?;
    //     // }
    // }

    // let dt = start_gen.elapsed();
    // println!(
    //     "\n{generated_tokens} tokens generated ({:.2} token/s)",
    //     generated_tokens as f64 / dt.as_secs_f64(),
    // );

    // // for (name, view) in tensors.tensors() {
    // //     let dtype = view.dtype();
    // //     let shape = view.shape();
    // //     println!("{name}: {dtype:?} - {shape:?}");
    // // }

    // // d_model: 768, <--------------------
    // // n_layer: 24, <---------------------
    // // vocab_size: 50277 <----------------
    // // pad_vocab_size_multiple: 8 <-------
    // // padded_vocab = (vocab_size + pad - 1) / pad * pad = (50277 + 7) / 8 * 8 = 6285 * 8 = 50280
    // // ssm_cfg: {},
    // // rms_norm: true,
    // // residual_in_fp32: true,
    // // fused_add_norm: true

    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
