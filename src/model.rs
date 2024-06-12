use std::marker::PhantomData;

use halo2_proofs::halo2curves::ff::{FromUniformBytes, PrimeField};

// pub struct ModelCircuit;

#[derive(Clone, Debug, Default)]
pub struct ModelCircuit<F: PrimeField> {
    //   pub used_gadgets: Arc<BTreeSet<GadgetType>>,
    //   pub dag_config: DAGLayerConfig,
    pub layers: Vec<Layer<F>>,
    pub tensors: BTreeMap<i64, Array<F, IxDyn>>,
    pub k: usize,
    pub bits_per_elem: usize,
    pub inp_idxes: Vec<i64>,
    pub num_random: i64,
}

// #[derive(Clone, Debug)]
// pub struct ModelConfig<F: PrimeField + Ord + FromUniformBytes<64>> {
// //   pub gadget_config: Rc<GadgetConfig>,
// //   pub public_col: Column<Instance>,
// //   pub _marker: PhantomData<F>,
// }
