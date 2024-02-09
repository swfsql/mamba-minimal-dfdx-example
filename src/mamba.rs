//! Utilizes dfdx's MambaBlock and other Modules to build a Mamba model capable of utilizing the state-spaces/mamba-130m text prediction models.
//!
//! References:
//! - https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/
//! - https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/

use dfdx::prelude::*;

// exports
pub use types::*;

/// The dimensions are set to be runtime values.
/// This avoids needing to define many more bounds for the model.
pub mod types {
    use super::*;

    /// How many independent inferences or training instances to run.
    /// Each instance has an independent state, inputs and outputs, and so on.
    ///
    /// Note: all instances on the same batch must have the same lengths (sequence, vocab, etc).
    pub type Batch = usize;

    /// How many sequence points (timesteps) the instances have.
    ///
    /// Note: for stateful inference or training, the sequence is always assumed to be 1.
    pub type Sequence = usize;

    /// The vocab for the embedding layer. The input is assumed to be an usize, representing different classes.
    /// Eg. for a-z, vocab amounts to 26.
    pub type Vocab = usize;

    /// Hidden dimension, the actual input and output for each Mamba block.
    ///
    /// Note: The embedding layer maps from the Vocab into DModel.
    pub type DModel = usize;

    /// Latent state dimension (`N` in Algorithm 2 from the Mamba paper).
    ///
    /// Defaults to `16`.
    pub type DState = usize;

    /// Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
    /// Δ or delta: input-dependent step size.
    ///
    /// Defaults to `(DModel + 15) / 16`.
    pub type DtRank = usize;

    /// Defaults to `4`.
    pub type DConv = usize;

    /// DModel * expand (`D` in Algorithm 2 from the Mamba paper).
    ///
    /// Defaults to `d_model * 2`.
    pub type DInner = usize;

    /// A [MambaBlockConfig] set to runtime values.
    pub type MambaBlockDynConfig = MambaBlockConfig<DModel, DState, DtRank, DConv, DInner>;
    /// A [MambaBlock] set to runtime values.
    pub type MambaBlockDyn<E, D> = MambaBlock<DModel, DState, DtRank, DConv, DInner, E, D>;
}

#[derive(Default, Debug, Clone, CustomModule)]
#[built(Mamba)]
pub struct MambaConfig {
    #[module]
    pub embedding: EmbeddingConfig<Vocab, DModel>,
    #[module]
    pub layers: Vec<ResidualMambaBlockConfig>,
    // note: in here no result discards are made, which differs from the references.
    #[module]
    pub norm_f: LayerRMSNorm1DConfig<DModel>,
    // TODO: delete this layer? It's the same weights from the embedding.
    #[module]
    pub lm_head: LinearConfig<DModel, Vocab>,
}

#[derive(Debug, Clone, Default, CustomModule)]
#[built(ResidualMambaBlock)]
pub struct ResidualMambaBlockConfig {
    #[module]
    pub res: ResidualAdd<(LayerRMSNorm1DConfig<DModel>, MambaBlockDynConfig)>,
}

impl MambaConfig {
    /// ### Parameters
    /// - `padded_vocab_size`: If no pad is required, this should be considered the unpadded vocab size.
    /// If pad is required, this should be the result of `(unpadded_vocab_size + pad - 1) / pad * pad`.
    /// - `d_state`: Defaults to `16`.
    /// - `dt_rank`: Defaults to `(d_model + 15) / 16`.
    /// - `d_conv`: Defaults to `4`.
    /// - `d_inner`: Defaults to `d_model * 2`.
    pub fn new(
        n_layer: usize,
        padded_vocab_size: usize,
        d_model: DModel,
        d_state: Option<DState>,
        dt_rank: Option<DtRank>,
        d_conv: Option<DConv>,
        d_inner: Option<DInner>,
    ) -> Self {
        let d_state = d_state.unwrap_or(16);
        let dt_rank = dt_rank.unwrap_or((d_model + 15) / 16);
        let d_conv = d_conv.unwrap_or(4);
        let d_inner = d_inner.unwrap_or(d_model * 2);

        MambaConfig {
            embedding: EmbeddingConfig {
                vocab: padded_vocab_size,
                model: d_model,
            },
            layers: {
                let mut layers = Vec::with_capacity(n_layer);
                for _ in 0..n_layer {
                    let mamba_block =
                        MambaBlockConfig::new(d_model, d_state, dt_rank, d_conv, d_inner);
                    let norm = LayerRMSNorm1DConfig(d_model);
                    let residual = ResidualAdd((norm, mamba_block));
                    let layer = ResidualMambaBlockConfig { res: residual };
                    layers.push(layer);
                }
                layers
            },
            norm_f: LayerRMSNorm1DConfig(d_model),
            lm_head: LinearConfig {
                inp: d_model,
                out: padded_vocab_size,
            },
        }
    }
}

pub mod stateless {
    use super::*;

    /// The Input for [Mamba] (stateless).
    ///
    /// Also the vocab [Embedding] input. Note that the data type is set to `usize`.
    pub type VocabInput<D, T> = Tensor<(Batch, Sequence), usize, D, T>;

    /// The Output for [Mamba] (stateless).
    ///
    /// Also the [MambaBlock] (stateless) Input/Output. Each instance contains all of it's timesteps.
    pub type BlockInput<E, D, T> = Tensor<(Batch, Sequence, DModel), E, D, T>;

    // mamba
    impl<E: Dtype, D: Device<E>, T: Tape<E, D>> Module<VocabInput<D, T>> for Mamba<E, D>
    where
        Embedding<Vocab, DModel, E, D>: Module<VocabInput<D, T>, Output = BlockInput<E, D, T>>,
        Vec<ResidualMambaBlock<E, D>>: Module<BlockInput<E, D, T>, Output = BlockInput<E, D, T>>,
    {
        type Output = BlockInput<E, D, T>;
        fn try_forward(&self, x: VocabInput<D, T>) -> Result<Self::Output, Error> {
            let x = self.embedding.try_forward(x)?;
            let x = self.layers.try_forward(x)?;
            let x = self.norm_f.try_forward(x)?;
            let x = self.lm_head.try_forward(x)?;
            Ok(x)
        }
    }

    // residual connection
    impl<E: Dtype, D: Device<E>, T: Tape<E, D>> Module<BlockInput<E, D, T>> for ResidualMambaBlock<E, D>
    where
        (LayerRMSNorm1D<DModel, E, D>, MambaBlockDyn<E, D>):
            Module<BlockInput<E, D, T>, Output = BlockInput<E, D, T>>,
    {
        type Output = BlockInput<E, D, T>;
        fn try_forward(&self, x: BlockInput<E, D, T>) -> Result<Self::Output, Error> {
            let x = self.res.try_forward(x)?;
            Ok(x)
        }
    }
}

pub mod stateful {
    use super::*;

    /// The input for [Mamba] (stateful).
    ///
    /// Also the vocab [Embedding] input and a list of the last [StateCache].
    /// Each state cache is specific to a [MambaBlock] (stateful) layer.
    pub type VocabInputWithStates<E, D, T> = (VocabInput<D, T>, Vec<StateCache<E, D, T>>);

    /// The output for [Mamba] (stateful).
    pub type SingleOutputWithStates<E, D, T> = (SingleInput<E, D, T>, Vec<StateCache<E, D, T>>);

    /// The vocab [Embedding] input. Note that the data type is set to `usize`.
    pub type VocabInput<D, T> = Tensor<(Batch,), usize, D, T>;

    /// The [MambaBlock] (stateful) Input/Output. Contains a [SingleInput] and the last [StateCache].
    pub type BlockInputWithState<E, D, T> = (SingleInput<E, D, T>, StateCache<E, D, T>);

    /// The new "single" input. Each instance contains only the last of it's timesteps.
    pub type SingleInput<E, D, T> = Tensor<(Batch, DModel), E, D, T>;

    /// A [MambaStateCache] set to runtime values.
    pub type StateCache<E, D, T> = MambaStateCache<Batch, DState, DConv, DInner, E, D, T>;

    // mamba
    impl<E: Dtype, D: Device<E>, T: Tape<E, D>> Module<VocabInputWithStates<E, D, T>> for Mamba<E, D>
    where
        Embedding<Vocab, DModel, E, D>: Module<VocabInput<D, T>, Output = SingleInput<E, D, T>>,
        ResidualMambaBlock<E, D>:
            Module<BlockInputWithState<E, D, T>, Output = BlockInputWithState<E, D, T>>,
    {
        type Output = SingleOutputWithStates<E, D, T>;
        #[allow(clippy::type_complexity)]
        fn try_forward(&self, x: VocabInputWithStates<E, D, T>) -> Result<Self::Output, Error> {
            let (x, states): (
                VocabInput<D, T>,
                Vec<MambaStateCache<Batch, DState, DConv, DInner, E, D, T>>,
            ) = x;

            let mut x: SingleInput<E, D, T> = self.embedding.try_forward(x)?;

            assert_eq!(self.layers.len(), states.len());
            let mut new_states = vec![];
            for (layer, state) in self.layers.iter().zip(states.into_iter()) {
                let (new_x, new_state) = layer.try_forward((x, state))?;
                new_states.push(new_state);
                x = new_x;
            }

            let x: SingleInput<E, D, T> = self.norm_f.try_forward(x)?;
            let x: SingleInput<E, D, T> = self.lm_head.try_forward(x)?;

            Ok((x, new_states))
        }
    }

    // residual connection
    impl<E: Dtype, D: Device<E>, T: Tape<E, D>> Module<BlockInputWithState<E, D, T>>
        for ResidualMambaBlock<E, D>
    where
        MambaBlockDyn<E, D>:
            Module<BlockInputWithState<E, D, T>, Output = BlockInputWithState<E, D, T>>,
    {
        type Output = BlockInputWithState<E, D, T>;
        fn try_forward(&self, x: BlockInputWithState<E, D, T>) -> Result<Self::Output, Error> {
            let (x, state) = x;
            let (norm, mamba_block) = &self.res.0;
            let x2: SingleInput<E, D, T> = norm.try_forward(x.with_empty_tape())?;
            let (x2, state) = mamba_block.try_forward((x2, state))?;
            let x: SingleInput<E, D, T> = x.try_add(x2)?;
            Ok((x, state))
        }
    }
}

pub mod load {
    use std::collections::HashMap;

    #[allow(clippy::useless_format)]
    pub fn load_renames(n_layer: usize) -> HashMap<String, String> {
        let mut load_renames = vec![];
        load_renames.push((
            format!("embedding.weight"),
            format!("backbone.embedding.weight"),
        ));
        for i in 0..n_layer {
            let ki = format!("layers.{i}");
            let vi = format!("backbone.layers.{i}");
            load_renames.push((
                format!("{ki}.res.0.1.in_proj.weight"),
                format!("{vi}.mixer.in_proj.weight"),
            ));
            load_renames.push((
                format!("{ki}.res.0.1.conv1d.weight"),
                format!("{vi}.mixer.conv1d.weight"),
            ));
            load_renames.push((
                format!("{ki}.res.0.1.conv1d_bias.bias"),
                format!("{vi}.mixer.conv1d.bias"),
            ));
            load_renames.push((
                format!("{ki}.res.0.1.x_proj.weight"),
                format!("{vi}.mixer.x_proj.weight"),
            ));
            load_renames.push((
                format!("{ki}.res.0.1.dt_proj.weight"),
                format!("{vi}.mixer.dt_proj.weight"),
            ));
            load_renames.push((
                format!("{ki}.res.0.1.dt_proj.bias"),
                format!("{vi}.mixer.dt_proj.bias"),
            ));
            load_renames.push((format!("{ki}.res.0.1.a_log"), format!("{vi}.mixer.A_log")));
            load_renames.push((format!("{ki}.res.0.1.d"), format!("{vi}.mixer.D")));
            load_renames.push((
                format!("{ki}.res.0.1.out_proj.weight"),
                format!("{vi}.mixer.out_proj.weight"),
            ));
            load_renames.push((format!("{ki}.res.0.0.gamma"), format!("{vi}.norm.weight")));
        }
        load_renames.push((format!("norm_f.gamma"), format!("backbone.norm_f.weight")));
        load_renames.push((
            format!("lm_head.weight"),
            format!("backbone.embedding.weight"),
        ));

        load_renames.into_iter().collect()
    }
}
