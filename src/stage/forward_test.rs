#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    use crate::{
        graph::Graph,
        numeric::NumericConfig,
        prover::prover_kzg::{KZGProver, StageType},
        stage::forward::ForwardCircuit,
        utils::{
            helpers::{configure_static, configure_static_numeric_config_default, to_field},
            loader::load_from_json,
        },
    };

    type F = Fr;

    #[test]
    fn test_forward() {
        // Config default numeric config
        let numeric_config = configure_static(NumericConfig {
            commitment: false,
            ..configure_static_numeric_config_default()
        });

        // Load test graph
        let graph = Graph::construct(
            load_from_json("src/utils/test.json"),
            numeric_config.scale_factor,
        );
        let mut circuit = ForwardCircuit::<F>::construct(graph.clone(), &numeric_config);

        // Set numeric config
        configure_static(NumericConfig {
            used_numerics: Arc::new(circuit.clone().used_numerics.clone()),
            commitment: false,
            ..numeric_config
        });

        // Run the circuit
        let score = circuit.run().unwrap();
        println!("score: {:?}", score);
        for (i, o) in score.iter().enumerate() {
            println!(
                "score[{}]: {}({})",
                i,
                o,
                *o as f64 / numeric_config.scale_factor as f64
            );
        }
        let public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();

        // Create proof
        let mut prover = KZGProver::construct(circuit.k);
        prover.set_forward(circuit.clone());
        let (pk, _vk) = prover.keygen_forward();
        let proof = prover.prove(StageType::Forward, &pk, public.clone());
        println!("sizeof(proof): {}", std::mem::size_of_val(&proof));

        // Mock prove
        let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
