use crate::{hf, mamba};
use crate::{LogitsProcessorWrapper, MambaWrapper};
use dfdx::prelude::*;
use hf_hub::{
    api::wasm::Api,
    types::{FilePath, RepoId, RevisionPath},
    Repo, RepoType,
};

pub async fn run() -> anyhow::Result<()> {
    let api = Api::new().await?;

    let mut timing = web_time::Instant::now();
    let tokenizer_filename = api
        .model(RepoId(hf::tokenizer::REPO_ID.into()))
        .get(&FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()))
        .await?;
    log::info!(
        "finished downloading/checking tokenizer in {}ms",
        timing.elapsed().as_millis()
    ); // 4s/2s

    timing = web_time::Instant::now();
    let repo = api.repo(Repo::with_revision(
        RepoId(hf::mamba_130m::REPO_ID.into()),
        RepoType::Model,
        RevisionPath(hf::mamba_130m::REVISION_PATH.into()),
    ));
    let mamba_filename = repo
        .get(&FilePath(
            hf::mamba_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
        ))
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
    let mamba = {
        let n_layer = 24;
        let padded_vocab_size = 50280;
        let d_model = 768;

        timing = web_time::Instant::now();
        log::info!("initializing random mamba model");
        let mamba =
            mamba::MambaConfig::new(n_layer, padded_vocab_size, d_model, None, None, None, None);
        let mut m: mamba::Mamba<f32, Cpu> = cpu.try_build_module::<f32>(mamba)?;
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

    let mut models = MambaWrapper::new(tokenizer, mamba);
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);

    let prompt = "Mamba is the";
    let sample_len = 200;
    let mut output = String::new();

    log::info!("Running mamba model");
    timing = web_time::Instant::now();
    let mut last_elapsed = timing.elapsed().as_millis();
    // stateful
    let mut i: usize = 0;
    {
        let (mut tokens, eos_token) = models.reset_prompt(prompt)?;

        // gets first token (as if it were an implicit output)
        if let Some(t) = tokens.first() {
            if let Some(t) = models.tokenizer.next_token(*t)? {
                output += &t;
            }
        }

        // initial states
        let mut states = models.empty_states()?;

        while i < sample_len {
            let this_elapsed = timing.elapsed().as_millis();
            if this_elapsed > last_elapsed + 1000 {
                last_elapsed = this_elapsed;
                log::info!("(generation still running..): {output}");
            }

            let next_logits = models.step(tokens[i], &mut states)?;
            let next_token = processor.add_logits(i, &mut tokens, next_logits)?;
            if next_token == eos_token {
                break;
            }

            // if the token has some valid representation, print it
            if let Some(t) = models.tokenizer.next_token(next_token)? {
                output += &t;
            }

            i += 1;
        }
        if let Some(rest) = models.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            output += &rest;
        }
    }
    let elapsed = timing.elapsed().as_millis();
    log::info!(
        "mamba model generated {} tokens in {}ms ({} token/s)",
        i,
        elapsed,
        (i * 1000) as f32 / elapsed as f32
    );
    log::info!("{output}");

    // models.run_stateless("Mamba is the", 14, &mut processor)?;
    // println!();
    // models.run_stateful("Mamba is the", 5000, &mut processor)?;
    // println!();

    Ok(())
}
