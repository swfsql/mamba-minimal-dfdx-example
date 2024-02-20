import init, {
    wasm_main,
} from "./pkg/mamba_minimal_dfdx_example.js";

async function run() {
    await init();
    wasm_main();
}
run();