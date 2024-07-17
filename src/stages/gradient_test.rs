#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
    use ndarray::Array;

    use crate::{
        loss::loss::LossType,
        numerics::numeric::NumericConfig,
        stages::gradient::GradientCircuit,
        utils::helpers::{configure_static, configure_static_numeric_config_default, to_field},
    };

    type F = Fr;

    #[test]
    fn test_gradient() {
        // config default numeric config
        let numeric_config = configure_static_numeric_config_default();

        let score = Array::from_shape_vec((1, 2), vec![300, 700])
            .unwrap()
            .into_dyn();
        let label = vec![0];
        let circuit = GradientCircuit::<F>::construct(score, label, LossType::SoftMax);

        // Set numeric config
        configure_static(NumericConfig {
            used_numerics: Arc::new(circuit.clone().used_numerics.clone()),
            ..numeric_config
        });

        // Run the circuit
        let (loss, gradient) = circuit.run().unwrap();
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
        let prover = MockProver::run(numeric_config.k as u32, &circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
