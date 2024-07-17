#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    use crate::{
        graph::Graph,
        numerics::numeric::NumericConfig,
        stages::forward::ForwardCircuit,
        utils::{
            helpers::{configure_static, configure_static_numeric_config_default, to_field},
            loader::load_from_json,
        },
    };

    type F = Fr;

    #[test]
    fn test_forward() {
        // config default numeric config
        let numeric_config = configure_static_numeric_config_default();

        let graph = Graph::construct(
            load_from_json("./utils/test.json"),
            numeric_config.scale_factor,
        );
        let mut circuit = ForwardCircuit::<F>::construct(graph.clone());

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

        // Verify the circuit
        let public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
