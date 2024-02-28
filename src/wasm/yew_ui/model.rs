use dfdx::tensor::Cpu;
use hf_hub::{
    api::wasm::{Api, ApiRepo, Metadata, UrlTemplate},
    types::{
        Endpoint, FilePath, FileUrl, RepoId, RevisionPath, TmpFileBlobKey, TmpFileBlobKeyList,
    },
    Repo, RepoType,
};
use tokenizers::Tokenizer;
use yew::prelude::*;

use crate::{
    hf, mamba, token_output_stream::TokenOutputStream, LogitsProcessorWrapper, MambaWrapper,
};

pub struct Model {
    // general data
    // TODO: allow to use wgpu device
    /// Dfdx [Cpu] device.
    pub device: Cpu,

    // fetching, loading, building
    /// Can check the cache, fetch and load data.
    pub cache_api: Connection<Api>,
    /// Stores cache and load status information, and also loaded bytes data.
    pub tokenizer: ModelData,
    /// Stores cache and load status information, and also loaded bytes data.
    pub mamba: ModelData,
    /// Consumes loaded bytes data to partially build the required models.
    pub models_wrapper_builder: MambaWrapperBuilder,

    // built models
    /// Models that are built and ready to use for inference.
    pub models_wrapper: Option<Wrapper>,

    // inference-related data
    /// Current user input.
    pub input: String,
    /// Whether the ongoing generation possibly no longer reflects the (new) user input.
    pub is_input_dirty: bool,
    pub is_reset: bool,
    pub is_generating: bool,
    pub generation_callback_interval: Option<gloo_timers::callback::Interval>,
    //
    /// Current token step index (for logits selection).
    pub step: usize,
    /// Tokens being (at first) introduced into or (later) produced by the generation.
    pub tokens: Vec<u32>,
    /// Current generation result (token concatenation from each generation step).
    pub output: String,
    /// The token the model uses to signal the end of the generation.
    pub eos_token: u32,
}

impl Model {
    pub fn select(&self, selection: &ModelSelection) -> &ModelData {
        match selection {
            ModelSelection::Tokenizer => &self.tokenizer,
            ModelSelection::Mamba => &self.mamba,
        }
    }

    pub fn select_mut(&mut self, selection: &ModelSelection) -> &mut ModelData {
        match selection {
            ModelSelection::Tokenizer => &mut self.tokenizer,
            ModelSelection::Mamba => &mut self.mamba,
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self {
            // general data
            device: Cpu::seed_from_u64(0),

            // fetching, loading, building
            cache_api: Connection::Disconnected,
            tokenizer: ModelData::new(
                "Tokenizer".into(),
                ModelDataConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(hf::tokenizer::REPO_ID.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath::default(),
                    filepath: FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()),
                }),
            ),
            mamba: ModelData::new(
                "Mamba-130m".into(),
                ModelDataConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(hf::mamba_130m::REPO_ID.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath(hf::mamba_130m::REVISION_PATH.into()),
                    filepath: FilePath(hf::mamba_130m::FILE_PATH_MODEL_SAFETENSORS.into()),
                }),
            ),
            models_wrapper_builder: MambaWrapperBuilder::default(),

            // built models
            models_wrapper: None,

            // inference-related data
            input: "Mamba is the".into(),
            is_input_dirty: false,
            is_reset: true,
            is_generating: false,
            generation_callback_interval: None,
            step: 0,
            tokens: vec![],
            output: "".into(),
            eos_token: 0,
        }
    }
}

#[derive(Default)]
pub struct MambaWrapperBuilder {
    pub tokenizer: Option<Tokenizer>,
    pub mamba: Option<mamba::Mamba<f32, Cpu>>,
}

impl MambaWrapperBuilder {
    pub fn is_ready(&self) -> bool {
        self.tokenizer.is_some() && self.mamba.is_some()
    }
    pub fn build(self) -> Wrapper {
        self.into()
    }
    // TODO: load in the background with webworkers
    pub fn with(&mut self, selection: &ModelSelection, data: Vec<u8>, device: &Cpu) {
        match selection {
            ModelSelection::Tokenizer => {
                let tokenizer = tokenizers::Tokenizer::from_bytes(data)
                    .map_err(anyhow::Error::msg)
                    .unwrap();
                self.tokenizer = Some(tokenizer);
            }
            ModelSelection::Mamba => {
                let mamba = {
                    let n_layer = 24;
                    let padded_vocab_size = 50280;
                    let d_model = 768;

                    // TODO: avoid random initialization and just initialize with zeroes
                    // TODO: avoid initialization and instead initialize directly from the data
                    log::info!("initializing random mamba model");
                    let mamba = mamba::MambaConfig::new(
                        n_layer,
                        padded_vocab_size,
                        d_model,
                        None,
                        None,
                        None,
                        None,
                    );
                    use dfdx::nn::BuildModuleExt;
                    let mut m: mamba::Mamba<f32, Cpu> =
                        device.try_build_module::<f32>(mamba).unwrap();
                    log::info!("random mamba model initialized"); // ~15-20s

                    let load_renames = mamba::load::load_renames(n_layer);
                    let skip_missing = true;
                    let mut key_map =
                        |key: String| load_renames.get(&key).unwrap_or(&key).to_string();

                    log::info!("loading mamba data");
                    use dfdx::nn::LoadSafeTensors;
                    m.load_safetensors_from_bytes_with(data.as_slice(), skip_missing, &mut key_map)
                        .unwrap();
                    log::info!("mamba data loaded");
                    m
                };
                self.mamba = Some(mamba);
            }
        }
    }
    pub fn merge(self, other: Self) -> Self {
        Self {
            tokenizer: self.tokenizer.or(other.tokenizer),
            mamba: self.mamba.or(other.mamba),
        }
    }
}

impl From<MambaWrapperBuilder> for Wrapper {
    fn from(value: MambaWrapperBuilder) -> Self {
        match (value.tokenizer, value.mamba) {
            (Some(t), Some(m)) => {
                let models = MambaWrapper::new(t, m);
                Wrapper::new(models)
            }
            (None, Some(_)) => panic!("missing tokenizer"),
            (Some(_), None) => panic!("missing mamba"),
            (None, None) => panic!("missing tokenizer and mamba"),
        }
    }
}

pub enum Connection<T> {
    Disconnected,
    Connecting,
    Connected(T),
    Disconnecting(T),
}

impl<T> Connection<T> {
    /// Note: not connected does not implies disconnected.
    pub fn is_exactly_connected(&self) -> bool {
        matches!(self, Self::Connected(_))
    }
    /// Note: not disconnected does not implies connected.
    pub fn is_exactly_disconnected(&self) -> bool {
        matches!(self, Self::Disconnected)
    }
    pub fn as_connected(&self) -> Option<&T> {
        if let Self::Connected(connected) = &self {
            Some(connected)
        } else {
            None
        }
    }
}

pub struct Wrapper {
    pub models: MambaWrapper,
    pub states: mamba::stateful::MambaStatesDyn<f32, dfdx::tensor::Cpu, dfdx::tensor::NoneTape>,
    pub processor: LogitsProcessorWrapper,
}

impl Wrapper {
    pub fn new(models: MambaWrapper) -> Self {
        let states = models.empty_states().unwrap();
        Self {
            models,
            states,
            processor: LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelData {
    pub label: String,
    pub config: ModelDataConfig,
    pub load: Load,
    pub cache: Cache,
}

impl ModelData {
    pub fn new(label: String, config: ModelDataConfig) -> Self {
        Self {
            label,
            config,
            load: Load::default(),
            cache: Cache::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelSelection {
    Tokenizer,
    Mamba,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelDataConfig {
    Huggingface(HuggingfaceConfig),
    Custom(CustomConfig),
}

impl ModelDataConfig {
    pub fn api_repo(&self, api: &Api) -> ApiRepo {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.api_repo(api),
        }
    }

    pub fn file_url(&self) -> FileUrl {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.file_url(),
        }
    }

    pub fn file_path(&self) -> &FilePath {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => &hf.filepath,
        }
    }

    pub async fn metadata(&self, api: &Api) -> Option<Metadata> {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.metadata(api).await,
        }
    }
    pub async fn check(&self, api: &Api, metadata: &Metadata) -> Option<TmpFileBlobKeyList> {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.check(api, metadata).await,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HuggingfaceConfig {
    endpoint: Endpoint,
    url_template: UrlTemplate,
    repo_id: RepoId,
    repo_type: RepoType,
    revision: RevisionPath,
    filepath: FilePath,
}

impl HuggingfaceConfig {
    pub fn api_repo(&self, api: &Api) -> ApiRepo {
        let repo = Repo::with_revision(self.repo_id.clone(), self.repo_type, self.revision.clone());
        api.repo(repo)
    }

    pub fn file_url(&self) -> FileUrl {
        let repo = Repo::with_revision(self.repo_id.clone(), self.repo_type, self.revision.clone());
        self.url_template
            .url(&self.endpoint, &repo, &self.revision, &self.filepath)
    }

    pub async fn metadata(&self, api: &Api) -> Option<Metadata> {
        let api_repo = self.api_repo(api);
        let file_url = api_repo.url(&self.filepath);
        let metadata = api.metadata(&file_url).await.unwrap();
        Some(metadata)
    }
    pub async fn check(&self, api: &Api, metadata: &Metadata) -> Option<TmpFileBlobKeyList> {
        let repo = Repo::new(self.repo_id.clone(), self.repo_type);
        let api_repo = api.repo(repo);
        let check = api_repo.check(&self.filepath, metadata).await.unwrap();
        Some(check)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CustomConfig {}

#[derive(Clone, Debug, PartialEq)]
pub struct Load {
    pub is_checking: bool,
    pub is_done: bool,
    pub is_busy: bool,
    pub data: Vec<u8>,
}

impl Default for Load {
    fn default() -> Self {
        Self {
            is_checking: true,
            is_busy: false,
            is_done: false,
            data: vec![],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cache {
    pub is_checking: bool,
    pub is_done: bool,
    pub is_busy: bool,
    pub fetching: CacheFetch,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            is_checking: true,
            is_done: Default::default(),
            is_busy: Default::default(),
            fetching: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheFetch {
    pub current_chunk: usize,
    pub metadata: Option<Metadata>,
    pub chunk_list: TmpFileBlobKeyList, // pub total_chunk: usize,
                                        // pub total_bytes: usize,
}
