#[cfg(test)]
mod tests {
    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    use crate::{
        graph::Graph,
        numeric::NumericConfig,
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
        let numeric_config = configure_static_numeric_config_default();

        // Load test graph
        let graph = Graph::construct(
            // load_from_json("src/utils/test.json"),
            load_from_json("examples/mlp2/model_lenet.json"),
            numeric_config.scale_factor,
            true,
        );
        let mut circuit = ForwardCircuit::<F>::construct(graph.clone());
        println!("graph:{:?}", graph);

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

        // Set numeric config used numerics
        println!("circuit.used_numerics: {:?}", circuit.used_numerics);
        let numeric_config = configure_static(NumericConfig {
            used_numerics: circuit.clone().used_numerics.clone().into(),
            ..numeric_config
        });

        let public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        println!("public: {:?}", public);

        // Mock prove
        let prover =
            MockProver::run(numeric_config.k as u32, &circuit, vec![public.clone()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        // Create proof
        // let mut prover =
    }
}
