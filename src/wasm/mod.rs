#[cfg(not(feature = "wasm_yew_ui"))]
pub mod non_ui;
#[cfg(feature = "wasm_yew_ui")]
pub mod yew_ui;

// TODO
// pub mod dioxus_ui;

use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub async fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Debug).unwrap();
    log::info!("wasm initialized");

    #[cfg(not(feature = "wasm_yew_ui"))]
    non_ui::run().await.unwrap();

    #[cfg(feature = "wasm_yew_ui")]
    {
        use crate::wasm::yew_ui::Msg;
        let handle = yew::Renderer::<yew_ui::Model>::new().render();
        handle.send_message_batch(vec![Msg::StartConnectApi]);
        // TODO: shouldn't the handle be awaited or something?
        // otherwise shouldn't the main thread drop on "wasm finished"?
        // wtf
    }

    log::info!("wasm finished");
}
