#[cfg(test)]
mod tests {
    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    use crate::{
        graph::Graph,
        loss::LossType,
        numeric::NumericConfig,
        stage::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
        utils::{
            helpers::{configure_static, configure_static_numeric_config_default, to_field},
            loader::load_from_json,
        },
    };

    type F = Fr;
    // backward test need to use forward and gradient
    #[test]
    fn test_backward() {
        // Config default numeric config
        let numeric_config = configure_static(NumericConfig {
            commitment: true,
            ..configure_static_numeric_config_default()
        });

        // Load test graph
        let graph = Graph::construct(
            load_from_json("src/utils/test.json"),
            numeric_config.scale_factor,
            true,
        );

        /* ------------ stage 1 forward ------------ */
        let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());

        // Set numeric config
        let numeric_config = configure_static(NumericConfig {
            used_numerics: forward_circuit.clone().used_numerics.clone(),
            ..numeric_config
        });

        // Run the circuit
        let score = forward_circuit.run().unwrap();
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
        let prover =
            MockProver::run(numeric_config.k as u32, &forward_circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        /* ------------ stage 2 gradient ------------ */
        let label = vec![0];
        let gradient_circuit = GradientCircuit::<F>::construct(score, label, LossType::SoftMax);

        // Set numeric config
        let numeric_config = configure_static(NumericConfig {
            used_numerics: gradient_circuit.clone().used_numerics.clone(),
            ..numeric_config
        });

        // Run the circuit
        let (loss, gradient) = gradient_circuit.run().unwrap();
        println!("loss: {:?}", loss);
        println!("gradient: {:?}", gradient);
        for (i, o) in gradient.iter().enumerate() {
            println!(
                "gradient[{}]: {}({})",
                i,
                o,
                *o as f64 / numeric_config.scale_factor as f64
            );
        }

        // Verify the circuit
        let public = gradient.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        let prover =
            MockProver::run(numeric_config.k as u32, &gradient_circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        /* ------------ stage 3 backward ------------ */
        let mut backward_graph = forward_circuit.graph.clone();
        backward_graph
            .tensor_map
            .insert("gradient".to_string(), gradient.clone());
        let mut backward_circuit = BackwardCircuit::<F>::construct(backward_graph.clone());

        // Set numeric config
        let numeric_config = configure_static(NumericConfig {
            used_numerics: backward_circuit.clone().used_numerics.clone(),
            ..numeric_config
        });

        let backward_gradient = backward_circuit.run().unwrap();
        println!("backward_gradient: {:?}", backward_gradient);

        // Verify the circuit
        let backward_public = backward_gradient
            .iter()
            .map(|x| to_field(*x))
            .collect::<Vec<_>>();
        let backward_prover = MockProver::run(
            numeric_config.k as u32,
            &backward_circuit,
            vec![backward_public],
        )
        .unwrap();
        assert_eq!(backward_prover.verify(), Ok(()));
        println!("Backward circuit verified");
    }
}
