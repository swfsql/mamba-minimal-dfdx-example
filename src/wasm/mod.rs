use core::slice::SlicePattern;

use crate::hf_mamba;
use crate::{hf_mamba::*, mamba, token_output_stream};
use candle_transformers::generation::LogitsProcessor;
use dfdx::prelude::*;
use hf_hub::{
    api::wasm::Api,
    types::{FilePath, RepoId, RevisionPath},
    Repo, RepoType,
};
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub async fn wasm_main() {
    // This hook is necessary to get panic messages in the console
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    // wasm_logger::init(wasm_logger::Config::default());
    console_log::init_with_level(log::Level::Debug).unwrap();
    log::info!("wasm initialized");
    wasm_main_inner().await.unwrap();
    log::info!("wasm finished");
}

async fn wasm_main_inner() -> anyhow::Result<()> {
    let api = Api::new().await?;

    let mut timing = web_time::Instant::now();
    let tokenizer_filename = api
        .model(RepoId(hf_mamba::TOKENIZER_MODEL_ID.into()))
        .get(&FilePath(hf_mamba::TOKENIZER_FILENAME.into()))
        .await?;
    log::info!(
        "finished downloading/checking tokenizer in {}ms",
        timing.elapsed().as_millis()
    ); // 4s/2s

    timing = web_time::Instant::now();
    let repo = api.repo(Repo::with_revision(
        RepoId(hf_mamba::MAMBA_MODEL_ID.into()),
        RepoType::Model,
        RevisionPath(hf_mamba::MAMBA_REVISION.into()),
    ));
    let mamba_filename = repo
        .get(&FilePath(hf_mamba::MAMBA_FILENAMES.into()))
        .await
        .unwrap();
    log::info!(
        "finished downloading/checking the mamba model in {}ms",
        timing.elapsed().as_millis()
    ); // ~180s/2s

    timing = web_time::Instant::now();
    log::info!("loading tokenizer data");
    let tokenizer = api.load_bytes(&tokenizer_filename).await;
    log::info!(
        "tokenizer data loaded in {}ms",
        timing.elapsed().as_millis()
    ); // ~100ms
    timing = web_time::Instant::now();
    log::info!("loading tokenizer");
    let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer).map_err(anyhow::Error::msg)?;
    log::info!("tokenizer loaded in {}ms", timing.elapsed().as_millis()); // ~200ms

    let cpu = Cpu::default();

    timing = web_time::Instant::now();
    log::info!("loading mamba data");
    let mamba_bytes = api.load_bytes(&mamba_filename).await;
    log::info!("mamba data loaded in {}ms", timing.elapsed().as_millis()); // ~2-3s
    let m = {
        let n_layer = 24;
        let padded_vocab_size = 50280;
        let d_model = 768;

        // TODO: avoid random initialization and just initialize with zeroes
        // TODO: avoid initialization and instead initialize directly from the data
        timing = web_time::Instant::now();
        log::info!("initializing random mamba model");
        let m =
            mamba::MambaConfig::new(n_layer, padded_vocab_size, d_model, None, None, None, None);
        let mut m: mamba::Mamba<f32, Cpu> = cpu.try_build_module::<f32>(m)?;
        log::info!(
            "random mamba model initialized in {}ms",
            timing.elapsed().as_millis()
        ); // ~15-20s

        timing = web_time::Instant::now();
        let load_renames = mamba::load::load_renames(n_layer);
        let skip_missing = true;
        let mut key_map = |key: String| load_renames.get(&key).unwrap_or(&key).to_string();
        log::info!("loading mamba");
        m.load_safetensors_from_bytes_with(mamba_bytes.as_slice(), skip_missing, &mut key_map)?;
        m
    };
    log::info!("mamba loaded in {}ms", timing.elapsed().as_millis()); // ~1s

    Ok(())
}
