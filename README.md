# mamba-minimal-dfdx-example

Adapted from [huggingface/candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/).

This is a temporary commit. Later this will be squashed etc.

### Building

##### Native
```bash
cargo run --release --features "native"
```

##### WASM
```bash
cargo watch -- wasm-pack build --release --target web --no-default-features
http -a 127.0.0.1
```

### Checking

##### Native
```bash
cargo check --features "native"
```

##### WASM
```bash
cargo check --target wasm32-unknown-unknown --lib --no-default-features
```

