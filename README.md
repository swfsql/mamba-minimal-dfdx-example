# mamba-minimal-dfdx-example

Adapted from [huggingface/candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/).

This is a temporary commit. Later this will be squashed etc.

### Building

##### Native
```bash
RUSTFLAGS="-C target-cpu=native"
cargo run --release --no-default-features --features "native"
```
TODO: add starting prompt as an arg parameter to the binary

##### WASM
```bash
# no-ui (web console only)
wasm-pack build --release --target web --no-default-features

# yew web ui
wasm-pack build --release --target web --no-default-features --features "wasm_yew_ui"

# serve
http -a 127.0.0.1
```
