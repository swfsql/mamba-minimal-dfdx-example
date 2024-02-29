use self::model::ModelSelection;
use hf_hub::{
    api::wasm::{Api, ApiError, Metadata},
    types::TmpFileBlobKeyList,
};
use indexed_db_futures::web_sys::DomException;
pub use model::{Connection, Model};
use yew::prelude::*;

pub mod model;
pub mod update;
pub mod view;

pub enum Msg {
    // Todo,

    // fetching, loading, building
    /// Starts the huggingface api connection (reqwest and indexeddb clients).
    StartConnectApi,
    /// Concludes the huggingface api connection (reqwest and indexeddb clients).
    FinishConnectApi(Api),
    FailConnectApi(ApiError),
    /// Starts the huggingface api disconnection (reqwest and indexeddb clients).
    StartDisconnectApi,
    /// Concludes the huggingface api disconnection (reqwest and indexeddb clients).
    FinishDisconnectApi,
    FailDisconnectApi,
    /// Starts checking information about the data of a model (size, etc).
    StartModelDataCheck(ModelSelection),
    /// Concludes checking information about the data of a model (size, etc).
    FinishModelDataCheck(ModelSelection, Metadata, TmpFileBlobKeyList),
    FailModelDataCheck,
    /// Starts fetching a model data.
    StartModelDataFetch(ModelSelection),
    /// Concludes fetching a single chunk of a model data.
    /// This is useful to state about the fetching progress.
    FinishModelDataFetchSingle(ModelSelection, usize),
    FailModelDataFetchSingle(ModelSelection, usize, ApiError),
    /// Concludes fetching a model data (all chunks).
    FinishModelDataFetch(ModelSelection),
    /// Starts uploading a model data.
    /// This is an alternative to the "fetch and cache read" mechanism.
    StartModelDataUpload(ModelSelection),
    /// Concludes uploading a model data.
    FinishModelDataUpload(ModelSelection),
    FailModelDataUpload,
    /// Starts loading (reading) a model data.
    /// The goal is to have bytes into the memory.
    StartModelDataLoad(ModelSelection),
    /// Concludes loading (reading) a model data.
    FinishModelDataLoad(ModelSelection, Vec<u8>),
    FailModelDataLoad(ModelSelection, DomException),
    /// Unloads a model data.
    /// The goal is to clear memory usage.
    /// If the model was built, it also get's unbuilt.
    /// Other models may also get unbuilt as a result of this action.
    ModelDataUnload(ModelSelection),
    /// Starts erasing a model data from the cache.
    /// The goal is to free HDD data.
    StartModelDataErase(ModelSelection),
    /// Concludes erasing a model data from the cache.
    FinishModelDataErase(ModelSelection),
    FailModelDataErase(ModelSelection, DomException),
    /// Starts building a model from the model data.
    /// This is when the data stops being raw bytes and become tensors (etc) instead.
    StartModelBuild(ModelSelection),
    /// Concludes building a model from the model data.
    FinishModelBuild(ModelSelection),
    FailModelBuild,
    /// If all required models are built, we move to the next step of being to use the models
    /// for inference (etc).
    TryFinilizeModelsBuilding,

    // user input
    /// What the user has as inserted to the input textarea.
    InputUpdate(String),

    // inference
    /// Starts the models inference.
    /// This can only be used from a zero (clean) initial state.
    StartGeneration,
    /// Ask for a single inference step.
    /// The goal is to avoid freezing the rendering by adding a small delay between the steps.
    StepGeneration,
    /// Stops (or pause) the models inference.
    StopGeneration,
    /// Resumes the models inference.
    /// The last cached states are used instead of a zero (clean) one.
    ResumeGeneration,
    /// Resets the last cached states into a zero (clean) one.
    ResetStates,
}

impl Component for model::Model {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self::default()
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        Model::view(self, ctx)
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        Model::update(self, ctx, msg)
    }
}
