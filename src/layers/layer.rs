use halo2_proofs::halo2curves::ff::PrimeField;


pub enum LayerType {
    FullyConnected,
    ReLU,
}

pub trait Layer<F: PrimeField> {
    
}