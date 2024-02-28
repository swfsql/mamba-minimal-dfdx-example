//! This is a copy or adaption from https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/.

// #![allow(clippy::erasing_op)]

use dfdx::prelude::*;
use hf_hub::types::FilePath;
use hf_hub::{
    api::sync::Api,
    types::{RepoId, RevisionPath},
    Repo, RepoType,
};
use mamba_minimal_dfdx_example::{hf, mamba, LogitsProcessorWrapper, MambaWrapper};

fn main() -> anyhow::Result<()> {
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let tokenizer_filename = api
        .model(RepoId(hf::tokenizer::REPO_ID.into()))
        .get(&FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()))?;
    println!(
        "tokenizer {} path: {tokenizer_filename:?}",
        hf::tokenizer::FILE_PATH_TOKENIZER_JSON
    );

    let repo = api.repo(Repo::with_revision(
        RepoId(hf::mamba_130m::REPO_ID.into()),
        RepoType::Model,
        RevisionPath(hf::mamba_130m::REVISION_PATH.into()),
    ));
    // let mamba_config_filename = repo.get(MAMBA_CONFIG)?;
    // println!("mamba {MAMBA_CONFIG} path: {mamba_config_filename:?}");
    let mamba_filename = repo.get(&FilePath(
        hf::mamba_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
    ))?;
    println!(
        "mamba {} path: {mamba_filename:?}",
        hf::mamba_130m::FILE_PATH_MODEL_SAFETENSORS
    );
    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let cpu = Cpu::default();

    let start = std::time::Instant::now();
    println!("started loading the model");
    let mamba = {
        let n_layer = 24;
        let padded_vocab_size = 50280;
        let d_model = 768;
        let m =
            mamba::MambaConfig::new(n_layer, padded_vocab_size, d_model, None, None, None, None);
        let mut m: mamba::Mamba<f32, Cpu> = cpu.try_build_module::<f32>(m)?;
        let load_renames = mamba::load::load_renames(n_layer);
        let skip_missing = true;
        let mut key_map = |key: String| load_renames.get(&key).unwrap_or(&key).to_string();
        m.load_safetensors_with(&mamba_filename, skip_missing, &mut key_map)?;
        m
    };
    println!("loaded the model in {:?}", start.elapsed());

    let mut models = MambaWrapper::new(tokenizer, mamba);
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);

    models.run_stateless("Mamba is the", 14, &mut processor)?;
    println!();
    models.run_stateful("Mamba is the", 5000, &mut processor)?;
    println!();

    Ok(())
}
