#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    use crate::{
        commitments::poseidon::PoseidonHash,
        graph::Graph,
        loss::loss::LossType,
        numerics::numeric::NumericConfig,
        stages::{backward::BackwardCircuit, forward::ForwardCircuit, gradient::GradientCircuit},
        utils::{
            helpers::{configure_static, configure_static_numeric_config_default, to_field},
            loader::load_from_json,
        },
        weight::{FieldWeight, Weight},
    };

    type F = Fr;
    // backward test need to use forward and gradient
    #[test]
    fn test_backward() {
        let lr = 1;

        // Config default numeric config
        let numeric_config = configure_static(NumericConfig {
            commitment: true,
            ..configure_static_numeric_config_default()
        });

        // Load test graph
        let graph = Graph::construct(
            load_from_json("src/utils/test.json"),
            numeric_config.scale_factor,
        );

        /* ------------ stage 1 forward ------------ */
        let mut forward_circuit = ForwardCircuit::<F>::construct(graph.clone());

        // Set numeric config
        let numeric_config = configure_static(NumericConfig {
            used_numerics: Arc::new(forward_circuit.clone().used_numerics.clone()),
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

        // Hash the weights
        let weight_hash = PoseidonHash::<F>::hash_vec_to_one(
            // Poseidon hash the forward weights
            FieldWeight::<F>::construct(
                forward_circuit.graph.nodes.clone(),
                forward_circuit.field_tensor_map.clone(),
            )
            .to_vec(),
        );

        // Verify the circuit
        let mut public = score.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
        public.push(weight_hash);
        let prover =
            MockProver::run(numeric_config.k as u32, &forward_circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        /* ------------ stage 2 gradient ------------ */
        let label = vec![0];
        let gradient_circuit = GradientCircuit::<F>::construct(score, label, LossType::SoftMax);

        // Set numeric config
        let numeric_config = configure_static(NumericConfig {
            used_numerics: Arc::new(gradient_circuit.clone().used_numerics.clone()),
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
        let mut backward_circuit = BackwardCircuit::<F>::construct(backward_graph.clone(), lr);

        // Set numeric config
        let numeric_config = configure_static(NumericConfig {
            used_numerics: Arc::new(backward_circuit.clone().used_numerics.clone()),
            ..numeric_config
        });

        let backward_gradient = backward_circuit.run().unwrap();
        println!("backward_gradient: {:?}", backward_gradient);

        // Hash the new weights
        let new_weight_hash = PoseidonHash::<F>::hash_vec_to_one(
            FieldWeight::<F>::construct_from_weight(Weight::construct(
                backward_circuit.graph.nodes.clone(),
                backward_circuit.graph.tensor_map.clone(),
            ))
            .to_vec(),
        );

        // Verify the circuit
        let mut backward_public = backward_gradient
            .iter()
            .map(|x| to_field(*x))
            .collect::<Vec<_>>();
        backward_public.push(weight_hash);
        backward_public.push(new_weight_hash);
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
