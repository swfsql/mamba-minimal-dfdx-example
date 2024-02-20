pub mod mamba;
pub mod token_output_stream;

use candle_transformers::generation::LogitsProcessor;
use dfdx::prelude::*;
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;

pub mod hf_mamba {
    pub const MAMBA_MODEL_ID: &str = "state-spaces/mamba-130m";
    pub const MAMBA_REVISION: &str = "refs/pr/1";
    pub const MAMBA_CONFIG: &str = "config.json";
    pub const MAMBA_FILENAMES: &str = "model.safetensors";
    pub const TOKENIZER_MODEL_ID: &str = "EleutherAI/gpt-neox-20b";
    pub const TOKENIZER_FILENAME: &str = "tokenizer.json";
}
use hf_mamba::*;

pub struct MambaWrapper {
    pub tokenizer: TokenOutputStream,
    pub mamba: mamba::Mamba<f32, Cpu>,
}

pub struct LogitsProcessorWrapper {
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl MambaWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn new(mamba: mamba::Mamba<f32, Cpu>, tokenizer: Tokenizer) -> Self {
        Self {
            mamba,
            tokenizer: TokenOutputStream::new(tokenizer),
        }
    }

    /// Clears the [Tokenizer] and returns the `prompt` as a list of Vocab tokens
    /// and also the eos token.
    pub fn reset_prompt(&mut self, prompt: &str) -> anyhow::Result<(Vec<u32>, u32)> {
        self.tokenizer.clear();
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        Ok((tokens, eos_token))
    }

    /// Initializes a list of empty (zero, null) [mamba::stateful::StateCache] for a stateful run.
    pub fn empty_states(
        &self,
    ) -> anyhow::Result<mamba::stateful::MambaStatesDyn<f32, Cpu, NoneTape>> {
        let cpu = self.mamba.embedding.weight.device();
        let mut states = vec![];
        for _ in 0..self.mamba.layers.len() {
            let state = cpu.try_build_module::<f32>(dfdx::nn::MambaStateCacheConfig::new(
                1,
                16,
                4,
                768 * 2,
            ))?;
            states.push(state);
        }
        Ok(states)
    }

    /// Reset and make up to `sample_len - 1` stateless calls to generate up to `sample_len - 1` tokens.
    pub fn run_stateless(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;
        let cpu = self.mamba.embedding.weight.device();

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut i = 0;
        'outer: while i < sample_len {
            let input = cpu
                .tensor_from_vec(tokens.clone(), (1, tokens.len()))
                .to_dtype::<usize>();
            let logits_list = <mamba::Mamba<f32, Cpu> as Module<
                mamba::stateless::VocabInput<Cpu, NoneTape>,
            >>::try_forward(&self.mamba, input)?;
            let shape = (logits_list.shape().2,);
            let logits_list = logits_list.as_vec();

            // logits contains an output for each timestep
            let logits_list = logits_list
                .chunks_exact(50280)
                .map(|chunk| {
                    candle_core::Tensor::from_slice(chunk, shape, &candle_core::Device::Cpu)
                })
                .skip(i)
                .collect::<Result<Vec<_>, _>>()?;
            for logits in logits_list.into_iter() {
                let next_token = logits_processor_config.add_logits(i, &mut tokens, logits)?;
                if next_token == eos_token {
                    break 'outer;
                }

                // if the token has some valid representation, print it
                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    use std::io::Write;
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                i += 1;
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        Ok(())
    }

    /// Reset and make up to `sample_len - 1` stateful calls to generate up to `sample_len - 1` tokens.
    pub fn run_stateful(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut states = self.empty_states()?;

        let mut i = 0;
        while i < sample_len {
            let next_logits = self.step(tokens[i], &mut states)?;
            let next_token = logits_processor_config.add_logits(i, &mut tokens, next_logits)?;
            if next_token == eos_token {
                break;
            }

            // if the token has some valid representation, print it
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                use std::io::Write;
                print!("{t}");
                std::io::stdout().flush()?;
            }

            i += 1;
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        Ok(())
    }

    /// Make a stateful call to generate a logits.
    ///
    /// `i` is the i-th call. For the first call, `i` should be `0`.
    pub fn step(
        &self,
        input: u32,
        states: &mut mamba::stateful::MambaStatesDyn<f32, Cpu, NoneTape>,
    ) -> anyhow::Result<candle_core::Tensor> {
        let cpu = self.mamba.embedding.weight.device();
        let input = cpu.tensor_from_vec(vec![input], (1,)).to_dtype::<usize>();
        let states_owned = std::mem::take(states);
        let input = (input, states_owned);
        let (logits, new_states) = <mamba::Mamba<f32, Cpu> as Module<
            mamba::stateful::VocabInputWithStates<f32, Cpu, NoneTape>,
        >>::try_forward(&self.mamba, input)?;
        *states = new_states;

        let shape = (logits.shape().1,);
        let logits =
            candle_core::Tensor::from_vec(logits.as_vec(), shape, &candle_core::Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?;
        Ok(logits)
    }
}

impl LogitsProcessorWrapper {
    pub fn new(
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        }
    }

    // TODO: check if `i` should really start a zero, or check if something else is wrong.
    //
    /// Add logits that represents a token.
    ///
    /// `i` is the i-th call. For the first call, `i` should be `0`.
    pub fn add_logits(
        &mut self,
        i: usize,
        tokens: &mut Vec<u32>,
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
            // let _discarded_token = logits_processor.sample(&logits)?;
        } else {
            // try to predict the next token
            next_token = self.logits_processor.sample(&logits)?;
            // add the token to the "tokens" list
            tokens.push(next_token);
            // *generated_tokens += 1;
        }
        Ok(next_token)
    }
}
