use super::model::ModelSelection;
pub use super::model::{self, Connection, Model};
use super::Msg;
use hf_hub::{api::wasm::Api, types::TmpFileBlobKey};
use yew::prelude::*;

const TICK_MILLIS: u32 = 1;

impl model::Model {
    pub fn update(&mut self, ctx: &Context<Self>, msg: Msg) -> bool {
        match msg {
            // Msg::Todo => {
            //     todo!()
            // }

            // fetching, loading, building
            Msg::StartConnectApi => {
                assert!(self.cache_api.is_exactly_disconnected());
                self.cache_api = Connection::Connecting;
                ctx.link().send_future(async {
                    match Api::new().await {
                        Ok(ok) => Msg::FinishConnectApi(ok),
                        Err(err) => Msg::FailConnectApi(err),
                    }
                });
                true
            }
            Msg::FinishConnectApi(api) => {
                self.cache_api = Connection::Connected(api);
                if self.tokenizer.cache.is_checking {
                    ctx.link()
                        .send_message(Msg::StartModelDataCheck(ModelSelection::Tokenizer));
                }
                if self.mamba.cache.is_checking {
                    ctx.link()
                        .send_message(Msg::StartModelDataCheck(ModelSelection::Mamba));
                }
                true
            }
            Msg::FailConnectApi(err) => {
                self.cache_api = Connection::Disconnected;
                log::error!("failed to connect to the api: {err}");
                true
            }
            Msg::StartDisconnectApi => {
                log::error!("Msg::StartDisconnectApi not yet implemented");
                false
            }
            Msg::FinishDisconnectApi => {
                log::error!("Msg::FinishDisconnectApi not yet implemented");
                false
            }
            Msg::FailDisconnectApi => {
                log::error!("Msg::FailDisconnectApi not yet implemented");
                false
            }
            Msg::StartModelDataCheck(selection) => {
                let model_data = self.select(&selection);
                let api = self.cache_api.as_connected().unwrap().clone();
                // assert that a lot of data won't be cloned
                assert!(model_data.load.data.is_empty());
                let model_data = model_data.clone();
                ctx.link().send_future(async move {
                    let metadata = model_data.config.metadata(&api).await.unwrap();
                    let chunk_list = model_data.config.check(&api, &metadata).await.unwrap();
                    Msg::FinishModelDataCheck(selection, metadata, chunk_list)
                });
                false
            }
            Msg::FinishModelDataCheck(selection, metadata, chunk_list) => {
                let model_data = self.select_mut(&selection);
                model_data.cache.is_checking = false;
                model_data.load.is_checking = false;
                model_data.cache.fetching.metadata = Some(metadata);
                model_data.cache.is_done = chunk_list.iter().all(|c| c.is_ok());
                model_data.cache.fetching.chunk_list = chunk_list;
                true
            }
            Msg::FailModelDataCheck => {
                todo!()
            }
            Msg::StartModelDataFetch(selection) => {
                let api = self.cache_api.as_connected().unwrap().clone();
                let model_data = self.select_mut(&selection);
                model_data.cache.is_busy = true;
                model_data.cache.fetching.current_chunk = 0;

                let chunk_files = model_data.cache.fetching.chunk_list.clone();
                let link = ctx.link().clone();
                let api_repo = model_data.config.api_repo(&api);
                let url = model_data.config.file_url();
                ctx.link().send_future(async move {
                    for (i, chunk_file) in chunk_files.into_iter().enumerate() {
                        if let Err(chunk_file) = chunk_file {
                            match api_repo.download_tempfile(&url, &chunk_file).await {
                                Ok(()) => {}
                                Err(err) => {
                                    return Msg::FailModelDataFetchSingle(selection, i, err)
                                }
                            }
                        }
                        link.send_message(Msg::FinishModelDataFetchSingle(selection, i));
                    }
                    Msg::FinishModelDataFetch(selection)
                });
                true
            }
            Msg::FinishModelDataFetchSingle(selection, i) => {
                let model_data = self.select_mut(&selection);
                model_data.cache.fetching.current_chunk += 1;
                let item = &mut model_data.cache.fetching.chunk_list[i];
                if item.is_err() {
                    *item = Ok(item.clone().unwrap_err());
                }
                true
            }
            Msg::FailModelDataFetchSingle(selection, i, err) => {
                log::error!("failed to fetch chunk {i} for {selection:?}; err: {err}");
                let model_data = self.select_mut(&selection);
                model_data.cache.is_busy = false;
                model_data.cache.is_done = false;
                true
            }
            Msg::FinishModelDataFetch(selection) => {
                let model_data = self.select_mut(&selection);
                model_data.cache.is_busy = false;
                model_data.cache.is_done = true;
                true
            }
            Msg::StartModelDataUpload(_selection) => {
                log::error!("Msg::StartModelDataUpload not yet implemented");
                false
            }
            Msg::FinishModelDataUpload(_selection) => {
                log::error!("Msg::FinishModelDataUpload not yet implemented");
                false
            }
            Msg::FailModelDataUpload => {
                log::error!("Msg::FailModelDataUpload not yet implemented");
                false
            }
            Msg::StartModelDataLoad(selection) => {
                let api = self.cache_api.as_connected().unwrap().clone();
                let model_data = self.select_mut(&selection);
                model_data.load.is_busy = true;
                let chunks_keys: Vec<TmpFileBlobKey> = model_data
                    .cache
                    .fetching
                    .chunk_list
                    .iter()
                    .cloned()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                ctx.link().send_future(async move {
                    match api.load_bytes(&chunks_keys).await {
                        Ok(ok) => Msg::FinishModelDataLoad(selection, ok),
                        Err(err) => Msg::FailModelDataLoad(selection, err),
                    }
                });
                true
            }
            Msg::FinishModelDataLoad(selection, data) => {
                let model_data = self.select_mut(&selection);
                model_data.load.data = data;
                ctx.link().send_message(Msg::StartModelBuild(selection));
                false
            }
            Msg::FailModelDataLoad(selection, err) => {
                log::error!("failed to load data for {selection:?}; err: {err:?}");
                let model_data = self.select_mut(&selection);
                model_data.load.is_busy = false;
                true
            }
            Msg::ModelDataUnload(selection) => {
                // stop any in-progress generation
                if self.is_generating {
                    ctx.link().send_message_batch(vec![
                        Msg::StopGeneration,
                        Msg::ResetStates,
                        Msg::ModelDataUnload(selection),
                    ]);
                    return false;
                }

                // clear generation-related memory
                self.tokens.clear();
                self.tokens.shrink_to_fit();
                self.output.shrink_to_fit();

                // clear built models
                if self.models_wrapper.is_some() {
                    self.models_wrapper = None;
                    self.tokenizer.load.is_done = false;
                    self.mamba.load.is_done = false;
                }
                // in case the models weren't fully built yet,
                // clear build and data-load memory
                else if let ModelSelection::Tokenizer = selection {
                    self.models_wrapper_builder.tokenizer = None;
                    self.tokenizer.load.data.clear();
                    self.tokenizer.load.data.shrink_to_fit();
                    self.tokenizer.load.is_done = false;
                } else if let ModelSelection::Mamba = selection {
                    self.models_wrapper_builder.mamba = None;
                    self.mamba.load.data.clear();
                    self.mamba.load.data.shrink_to_fit();
                    self.mamba.load.is_done = false;
                };

                true
            }
            Msg::StartModelDataErase(selection) => {
                let api = self.cache_api.as_connected().unwrap().clone();
                let model_data = self.select_mut(&selection);
                assert!(!model_data.load.is_busy);
                model_data.cache.is_busy = true;
                let chunks_keys = model_data
                    .cache
                    .fetching
                    .chunk_list
                    .iter()
                    .cloned()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();

                ctx.link().send_future(async move {
                    match api.delete_bytes(&chunks_keys).await {
                        Ok(()) => Msg::FinishModelDataErase(selection),
                        Err(err) => Msg::FailModelDataErase(selection, err),
                    }
                });

                model_data.cache.is_busy = true;
                true
            }
            Msg::FinishModelDataErase(selection) => {
                let model_data = self.select_mut(&selection);
                model_data.cache.is_busy = false;
                model_data.cache.is_done = false;

                // set chunks to uncached
                for chunk_key in model_data.cache.fetching.chunk_list.iter_mut() {
                    let owned = chunk_key.clone().unwrap();
                    *chunk_key = Err(owned);
                }

                true
            }
            Msg::FailModelDataErase(selection, err) => {
                log::error!("failed to erase data for {selection:?}; err: {err:?}");
                let model_data = self.select_mut(&selection);
                model_data.cache.is_busy = false;
                model_data.cache.is_done = false;
                // re-check if the data is cached or not
                //
                // note: if partial data has been erased, then the button will switch to
                // the "click to fetch" option.
                model_data.cache.is_checking = true;
                ctx.link().send_message(Msg::StartModelDataCheck(selection));
                true
            }
            Msg::StartModelBuild(selection) => {
                let model_data = self.select_mut(&selection);
                let data = std::mem::take(&mut model_data.load.data);
                self.models_wrapper_builder
                    .with(&selection, data, &self.device);
                ctx.link().send_message(Msg::FinishModelBuild(selection));
                false
            }
            Msg::FinishModelBuild(selection) => {
                let model_data = self.select_mut(&selection);
                model_data.load.is_busy = false;
                model_data.load.is_done = true;
                ctx.link().send_message(Msg::TryFinilizeModelsBuilding);
                true
            }
            Msg::FailModelBuild => {
                todo!()
            }
            Msg::TryFinilizeModelsBuilding => {
                // consume the built models if they are all ready
                if self.models_wrapper_builder.is_ready() {
                    let builder = std::mem::take(&mut self.models_wrapper_builder);
                    let wrapper = builder.build();
                    self.models_wrapper = Some(wrapper);
                }
                true
            }

            // user input
            Msg::InputUpdate(new) => match (self.input == new, self.is_reset) {
                (_same @ true, _reset) => false,
                (_same @ false, _reset @ false) => {
                    self.input = new;
                    true
                }
                (_same @ false, _reset @ true) => {
                    self.input = new;
                    false
                }
            },

            // inference
            Msg::StartGeneration => {
                assert!(self.is_reset);
                assert!(!self.is_input_dirty);
                assert!(!self.is_generating);
                self.is_generating = true;
                self.is_reset = false;
                self.output.clear();
                self.step = 0;
                let models_wrapper = self.models_wrapper.as_mut().unwrap();
                let (tokens, eos_token) = models_wrapper.models.reset_prompt(&self.input).unwrap();
                self.tokens = tokens;
                self.eos_token = eos_token;

                // gets first token (as if it were an implicit output)
                if let Some(t) = self.tokens.first() {
                    if let Some(t) = models_wrapper.models.tokenizer.next_token(*t).unwrap() {
                        self.output += &t;
                    }
                }

                let link = ctx.link().clone();
                let interval = gloo_timers::callback::Interval::new(TICK_MILLIS, move || {
                    link.send_message(Msg::StepGeneration)
                });
                self.generation_callback_interval = Some(interval);

                true
            }

            Msg::StepGeneration => {
                if !self.is_generating {
                    return true;
                }
                let models_wrapper = self.models_wrapper.as_mut().unwrap();
                let models = &mut models_wrapper.models;
                let next_logits = models
                    .step(self.tokens[self.step], &mut models_wrapper.states)
                    .unwrap();
                let next_token = models_wrapper
                    .processor
                    .add_logits(self.step, &mut self.tokens, next_logits)
                    .unwrap();

                // if the token has some valid representation, print it
                if let Some(t) = models.tokenizer.next_token(next_token).unwrap() {
                    self.output += &t;
                }
                self.step += 1;

                if next_token == self.eos_token {
                    self.is_generating = false;

                    if let Some(rest) = models
                        .tokenizer
                        .decode_rest()
                        .map_err(anyhow::Error::msg)
                        .unwrap()
                    {
                        self.output += &rest;
                    }

                    true
                } else {
                    true
                }
            }
            Msg::StopGeneration => {
                self.is_generating = false;
                self.generation_callback_interval = None;
                true
            }
            Msg::ResumeGeneration => {
                assert!(!self.is_input_dirty);
                self.is_generating = true;
                let link = ctx.link().clone();
                let interval = gloo_timers::callback::Interval::new(TICK_MILLIS, move || {
                    link.send_message(Msg::StepGeneration)
                });
                self.generation_callback_interval = Some(interval);
                true
            }
            Msg::ResetStates => {
                assert!(!self.is_generating);
                let models_wrapper = self.models_wrapper.as_mut().unwrap();
                models_wrapper.states = models_wrapper.models.empty_states().unwrap();
                // for state in models_wrapper.states.iter_mut() {
                //     ResetParams::<f32, Cpu>::try_reset_params(state).unwrap();
                // }
                models_wrapper.processor =
                    crate::LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
                self.is_reset = true;
                self.is_input_dirty = false;
                true
            }
        }
    }
}
