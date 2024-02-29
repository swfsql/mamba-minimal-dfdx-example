use super::model::ModelSelection;
pub use super::model::{self, Connection, Model};
use super::Msg;
use crate::wasm::yew_ui::model::ModelData;
use yew::prelude::*;

impl model::Model {
    pub fn view(&self, ctx: &Context<Self>) -> Html {
        let link = ctx.link();

        let navbar = html_nested! {
            <nav
                class="navbar has-background-light is-fixed-bottom"
                role="navigation"
                aria_label="main navigation"
            >
            <div class="navbar-brand"/>
            <div class="navbar-menu">
                <div class="navbar-end">
                    <div class="navbar-item">
                        if self.cache_api.is_exactly_connected() {
                            <span class="tag is-success is-light">
                                {"Connected to cache"}
                            </span>
                        } else if self.cache_api.is_exactly_disconnected() {
                            <span class="tag is-danger is-light">
                                {"Disconnected from cache"}
                            </span>
                        } else {
                            <span class="tag is-loading is-light">
                                {"Awaiting.."}
                            </span>
                        }
                    </div>
                </div>
            </div>
            </nav>
        };
        let input = html_nested! {
            <div class="tile is-child">
            <label class="label">{"Your Input"}</label>
            <textarea
                class="textarea"
                name="user-input"
                rows="4"
                placeholder="Insert here what the model should start with.."
                value={self.input.clone()}
                oninput={link.callback(|e: InputEvent| Msg::InputUpdate(value_from_event(e)))}
            />
            <label class="help">{"TODO"}</label>
            </div>
        };
        let output = html_nested! {
            <div class="tile is-child">
            <label class="label">{"Generated Content"}</label>
            <textarea
                class="textarea"
                name="mamba-output"
                rows="8"
                placeholder="The continuation prediction will appear in here.."
                value={self.output.clone()}
            />
            <label class="help">{"TODO"}</label>
            </div>
        };
        let controls = html_nested! {
            <div class="tile is-parent">
            <div class="tile is-child">
                if self.models_wrapper.is_some() {
                    if self.is_generating {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::StopGeneration)}
                            >
                            {"Pause Generation"}
                        </button>
                        <label class="help">{"Click to pause generation"}</label>
                    } else if self.is_reset {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::StartGeneration)}
                            >
                            {"Start Generation"}
                        </button>
                        <label class="help">{"Click to start generation"}</label>
                    } else {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::ResumeGeneration)}
                            >
                            {"Resume Generation"}
                        </button>
                        <label class="help">{"Click to resume generation"}</label>
                    }
                } else {
                    <button
                        class="button"
                        disabled={true}
                        >
                        {"Start Generation"}
                    </button>
                    <label class="help">{"Models not yet loaded"}</label>
                }
            </div>
            <div class="tile is-child">
                if self.models_wrapper.is_some() {
                    if self.is_generating {
                        <button
                            class="button"
                            disabled={true}
                            >
                            {"Reset States"}
                        </button>
                        <label class="help">{"Generation in progress"}</label>
                    } else if self.is_reset {
                        <button
                            class="button"
                            // disabled={true}
                            >
                            {"Reset States"}
                        </button>
                        <label class="help">{"Already reset"}</label>
                    } else {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::ResetStates)}
                            >
                            {"Reset States"}
                        </button>
                        <label class="help">{"Click to reset states"}</label>
                    }
                } else {
                    <button
                        class="button"
                        disabled={true}
                        >
                        {"Reset States"}
                    </button>
                    <label class="help">{"Models not yet loaded"}</label>
                }
            </div>
            </div>
        };

        let tokenizer_model_data = model_data(link, &self.tokenizer, ModelSelection::Tokenizer);
        let mamba_model_data = model_data(link, &self.mamba, ModelSelection::Mamba);

        let caches = html_nested! {
            <div class="tile is-child is-vertical">
            {tokenizer_model_data}
            {mamba_model_data}
            </div>
        };
        let section = html_nested! {
            <section class="section">
            <div class="container">
                <div class="tile is-ancestor">
                    <div class="tile is-vertical">
                        <div class="tile is-parent is-vertical">
                            <div class="tile is-child">
                                <h2 class="subtitle">{"Content"}</h2>
                            </div>
                            {input}
                            {output}
                            {controls}
                        </div>
                        <div class="tile is-vertical">
                            <div class="tile is-child">
                            <h2 class="subtitle">{"Model Data"}</h2>
                            </div>
                            {caches}
                        </div>
                    </div>
                </div>
            </div>
            </section>
        };

        html! {<>
            {navbar}
            {section}
        </>
        }
    }
}

fn model_data<M: yew::Component<Message = Msg>>(
    link: &yew::html::Scope<M>,
    model_data: &ModelData,
    selection: ModelSelection,
) -> Html {
    let label = &model_data.label;
    let cache = &model_data.cache;
    let load = &model_data.load;
    let total_bytes = cache
        .fetching
        .metadata
        .as_ref()
        .map(|md| md.size)
        .unwrap_or_default();
    let total_bytes_human = &humansize::format_size(total_bytes, humansize::DECIMAL);
    let current_chunk = cache.fetching.current_chunk;
    let total_chunks = cache.fetching.chunk_list.len();
    let file_url = model_data.config.file_url().0;
    let file_path = &model_data.config.file_path().0;
    let data_load = match (load.is_checking, load.is_done, load.is_busy) {
        (_checking @ true, _, _) => html_nested! {<>
            <button class="button is-outline is-loading" disabled={true}>
                {"Checking"}
            </button>
            <label class="help">
                {"Checking.."}
            </label>
        </>
        },
        (_checking @ false, _loaded @ false, _loading @ false) => {
            if !cache.is_busy && cache.is_done {
                html_nested! {<>
                    <button
                        class="button is-danger"
                        onclick={link.callback(move |_| Msg::StartModelDataLoad(selection))}
                    >
                        {"Not loaded"}
                    </button>
                    <label class="help">
                        {"Click to load from cache"}
                    </label>
                </>
                }
            } else {
                html_nested! {<>
                    <button class="button is-outlined is-danger" disabled={true}>
                        {"Not loaded"}
                    </button>
                    <label class="help">
                        {"Please fetch or upload"}
                    </label>
                </>
                }
            }
        }
        (false, false, _loading @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Loading"}
            </button>
            <label class="help">
                {format!("Loading ({total_bytes_human})..")}
            </label>
        </>
        },
        (false, _loaded @ true, false) => html_nested! {<>
            <button
                class="button is-success is-light"
                onclick={link.callback(move |_| Msg::ModelDataUnload(selection))}
            >
                {format!("Loaded ({total_bytes_human})..")}
            </button>
            <label class="help">
                {"Click to unload"}
            </label>
        </>
        },
        // TODO: remove since the unload happens all at once before a re-render
        (false, _loaded @ true, _unloading @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Unloading"}
            </button>
            <label class="help">
                {format!("Unloading ({total_bytes_human})..")}
            </label>
        </>
        },
    };

    let data_cache = match (cache.is_checking, cache.is_done, cache.is_busy) {
        (_checking @ true, _, _) => html_nested! {<>
            <button class="button is-outline is-loading" disabled={true}>
                {"Checking"}
            </button>
            <label class="help">
                {"Checking.."}
            </label>
        </>
        },
        (_checking @ false, _cached @ false, _fetching @ false) => html_nested! {<>
            <button
                class="button"
                onclick={link.callback(move |_| Msg::StartModelDataFetch(selection))}
            >
                {"Not cached"}
            </button>
            <label class="help">
                {format!("Click to fetch ({total_bytes_human})")}
            </label>
        </>
        },
        (false, false, _fetching @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Fetching"}
            </button>
            <label class="help">
                {format!("Fetching {total_bytes_human} ({current_chunk}/{total_chunks})")}
            </label>
        </>
        },
        (false, _cached @ true, false) => html_nested! {<>
            <button
                class="button is-success is-light"
                onclick={link.callback(move |_| Msg::StartModelDataErase(selection))}
            >
                {format!("Cached ({total_bytes_human})")}
            </button>
            <label class="help">
                {"Click to erase from cache"}
            </label>
        </>
        },
        (false, _cached @ true, _erasing @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Erasing"}
            </button>
            <label class="help">
                {format!("Erasing {total_bytes_human}..")}
            </label>
        </>
        },
    };
    html_nested! {
        <div class="tile is-parent is-vertical">
        <div class="tile is-child">
            <label class="label">
                {label}
            </label>
        </div>
        <div class="tile is-parent">
            <div class="tile is-child">
                {data_load}
            </div>
            <div class="tile is-child">
                {data_cache}
            </div>
            <div class="tile is-child">
                <a
                    class="button"
                    href={file_url}
                    download={file_path.clone()}
                    target="_blank"
                >
                    {"Download to file"}
                </a>
                <label class="help">
                    {"Click to download or open in new tab"}
                </label>
            </div>
            <div class="tile is-child">
                <button
                    class="button"
                    disabled={true}
                    onclick={link.callback(move |_| Msg::StartModelDataUpload(selection))}
                >
                    {"Load from file"}
                </button>
                <label class="help">
                    {"TODO"}
                    // {"Click to upload"}
                </label>
            </div>
        </div>
        </div>
    }
}

pub fn value_from_event(e: InputEvent) -> String {
    use wasm_bindgen::JsCast;
    let event: Event = e.dyn_into().unwrap();
    let event_target = event.target().unwrap();
    let target: indexed_db_futures::web_sys::HtmlTextAreaElement = event_target.dyn_into().unwrap();
    target.value()
}
