pub mod assign;
pub mod backward;
pub mod backward_test;
pub mod forward;
pub mod forward_test;
pub mod gradient;
pub mod gradient_test;

use crate::{
    graph::{Graph, Node},
    loss::LossType,
    numeric::{NumericConfig, NumericType},
    operation::OPType,
    prover::{proof::Proof, prover::run_prove_kzg},
    utils::{
        helpers::{
            configure_static, get_numeric_config, to_field, to_primitive, update_graph, FieldTensor,
        },
        matcher::match_op_type,
        math::Int,
    },
};
use backward::BackwardCircuit;
use forward::ForwardCircuit;
use gradient::GradientCircuit;
use halo2_proofs::{
    dev::MockProver,
    halo2curves::{
        bn256::Fr,
        ff::{FromUniformBytes, PrimeField},
    },
};
use log::{debug, info};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPool,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    str::FromStr,
    thread::sleep,
    time::Duration,
};

pub trait CircuitName {
    fn name(&self) -> String;
}

type F = Fr;

// lazy_static! {
//     pub static ref NO_PROVE: bool = false;
// }
pub static mut NO_PROVE: bool = false;
// const NO_PROVE: bool = false;

pub struct Trainer {
    pub forward_k: u32,
    pub gradient_k: u32,
    pub backward_k: u32,
    pub graph: Graph,
    pub loss: Option<LossType>,
    pub f_stage: Option<Vec<(ForwardCircuit<F>, FieldTensor<F>, BTreeSet<NumericType>)>>,
    pub g_stage: Option<(GradientCircuit<F>, FieldTensor<F>, BTreeSet<NumericType>)>,
    pub b_stage: Option<Vec<(BackwardCircuit<F>, FieldTensor<F>, BTreeSet<NumericType>)>>,
    pub pool: ThreadPool,
    pub dir_path: PathBuf,
}

impl Trainer {
    pub fn new(
        forward_k: u32,
        gradient_k: u32,
        backward_k: u32,
        graph: Graph,
        loss: Option<LossType>,
        pool: ThreadPool,
        dir_path: Option<String>,
    ) -> Self {
        Self {
            forward_k,
            gradient_k,
            backward_k,
            graph,
            loss,
            f_stage: None,
            g_stage: None,
            b_stage: None,
            pool,
            dir_path: match dir_path {
                Some(path) => PathBuf::from_str(&path).unwrap(),
                None => PathBuf::from_str("./params").unwrap(),
            },
        }
    }

    fn split_graph_node(&self, graph: Graph) -> Vec<Vec<Node>> {
        // split graph based on "Conv" & "Gemm"
        let mut split_graph = vec![];
        let mut tmp = vec![];
        for node in graph.nodes.clone() {
            if (match_op_type(node.op_type.clone()) == OPType::Conv
                || match_op_type(node.op_type.clone()) == OPType::GEMM)
                && !tmp.is_empty()
            {
                split_graph.push(tmp);
                tmp = vec![];
            }
            tmp.push(node);
        }
        if !tmp.is_empty() {
            split_graph.push(tmp);
        }

        split_graph
    }

    fn configure(&self, used_numerics: BTreeSet<NumericType>) -> NumericConfig {
        configure_static(NumericConfig {
            used_numerics,
            ..get_numeric_config().clone()
        })
    }

    pub fn forward(&mut self, input: BTreeMap<String, FieldTensor<F>>) -> FieldTensor<F> {
        // Insert input tensor into graph
        for (key, value) in input.iter() {
            self.graph
                .tensor_map
                .insert(key.clone(), value.mapv(|x| to_primitive(&x)));
        }
        // Run forward non-circuit
        let (circuit, scores) = run_forward(self.graph.clone(), self.forward_k, true);

        self.graph = circuit.graph.clone();
        self.f_stage = Some(vec![(
            circuit.clone(),
            scores.clone(),
            circuit.used_numerics.clone(),
        )]);
        scores
    }

    pub fn forward_parallel(&mut self, input: BTreeMap<String, FieldTensor<F>>) -> FieldTensor<F> {
        // Insert input tensor into graph
        for (key, value) in input.iter() {
            self.graph
                .tensor_map
                .insert(key.clone(), value.mapv(|x| to_primitive(&x)));
        }
        // Run forward non-circuit
        let (circuit, scores) = run_forward(self.graph.clone(), self.forward_k, false);
        let used_numerics = circuit.used_numerics.clone();

        // split graph based on "Conv" & "Gemm"
        let split_graph = self.split_graph_node(self.graph.clone());

        // update graph
        self.graph = circuit.graph.clone();

        // split circuit // self.pool.install(|| {
        let f_stages = split_graph
                .par_iter()
                .map(|nodes| {
                    let (sub_graph, sub_outputs) = Graph::construct_sub_forward_graph(
                        nodes.clone(),
                        circuit.graph.tensor_map.clone(),
                    );
                    let sub_forward_circuit = ForwardCircuit::<F>::construct(sub_graph);
                    let sub_forward_public = sub_outputs.map(|x| to_field(*x));
                    info!(
                        "{:?} Mock Forward Circuit Created",
                        nodes
                            .iter()
                            .map(|node| node.op_type.clone())
                            .collect::<Vec<_>>()
                    );
                    (
                        sub_forward_circuit,
                        sub_forward_public,
                        used_numerics.clone(),
                    )
                })
                .collect()
        // })
        ;

        self.f_stage = Some(f_stages);
        scores
    }

    pub fn gradient(&mut self, scores: FieldTensor<F>, labels: Vec<Int>) -> FieldTensor<F> {
        let (circuit, gradient) = match self.loss {
            Some(loss) => run_gradient(scores.clone(), labels.clone(), loss, self.gradient_k, true),
            None => panic!("Loss function is not defined"),
        };
        self.g_stage = Some((
            circuit.clone(),
            gradient.clone(),
            circuit.used_numerics.clone(),
        ));
        gradient
    }

    pub fn backward(&mut self, gradient: FieldTensor<F>) -> FieldTensor<F> {
        let (circuit, backward) = run_backward(gradient, self.graph.clone(), self.backward_k, true);
        self.graph = circuit.graph.clone();
        self.b_stage = Some(vec![(
            circuit.clone(),
            backward.clone(),
            circuit.used_numerics.clone(),
        )]);
        backward
    }

    pub fn backward_parallel(&mut self, gradient: FieldTensor<F>) -> FieldTensor<F> {
        let (circuit, backward) =
            run_backward(gradient, self.graph.clone(), self.backward_k, false);
        let used_numerics = circuit.used_numerics.clone();

        // split graph based on "Conv" & "Gemm"
        let split_graph = update_graph(&circuit.graph, &self.graph.tensor_map);
        let split_nodes = self.split_graph_node(split_graph.clone());

        // update graph
        self.graph = circuit.graph.clone();

        // split circuit // self.pool.install(|| {
        let b_stages = split_nodes
                .par_iter()
                .map(|nodes| {
                    let (sub_graph, sub_outputs) = Graph::construct_sub_backward_graph(
                        nodes.clone(),
                        split_graph.tensor_map.clone(),
                    );
                    let sub_backward_circuit = BackwardCircuit::<F>::construct(sub_graph);
                    let sub_backward_public = sub_outputs.map(|x| to_field::<F>(*x));
                    info!(
                        "{:?} Mock Backward Circuit Verified",
                        nodes
                            .iter()
                            .map(|node| node.op_type.clone())
                            .collect::<Vec<_>>()
                    );
                    (
                        sub_backward_circuit,
                        sub_backward_public,
                        used_numerics.clone(),
                    )
                })
                .collect()
        // })
        ;

        self.b_stage = Some(b_stages);
        backward
    }

    pub fn train(
        &mut self,
        input: BTreeMap<String, FieldTensor<F>>,
        labels: Vec<Int>,
    ) -> FieldTensor<F> {
        let scores = self.forward(input);
        let gradient = self.gradient(scores.clone(), labels);
        let backward = self.backward(gradient);
        backward
    }

    pub fn train_parallel(
        &mut self,
        input: BTreeMap<String, FieldTensor<F>>,
        labels: Vec<Int>,
    ) -> FieldTensor<F> {
        let scores = self.forward_parallel(input);
        let gradient = self.gradient(scores.clone(), labels);
        let backward = self.backward_parallel(gradient);
        backward
    }

    pub fn prove_forward(&self) -> Vec<Proof> {
        if unsafe { NO_PROVE } {
            sleep(Duration::from_secs(5));
            return vec![Default::default()];
        }
        let start = std::time::Instant::now();
        let proofs = match &self.f_stage {
            Some(f_stages) => f_stages
                .par_iter()
                // .iter()
                .enumerate()
                .map(|(i, (circuit, public, used_numerics))| {
                    let numeric_config = self.configure(used_numerics.clone());
                    let (_, proof) = run_prove_kzg(
                        self.forward_k,
                        Some(self.dir_path.join("forward").join(i.to_string())),
                        circuit.clone(),
                        public.clone(),
                        numeric_config.assigned_num_cols,
                        circuit.commitment_tuples(),
                    );

                    proof
                })
                .collect::<Vec<_>>(),
            None => panic!("Forward circuit is not initialized"),
        };
        info!("Prove Forward Time: {:?}", start.elapsed());
        proofs
    }

    pub fn prove_gradient(&self) -> Vec<Proof> {
        if unsafe { NO_PROVE } {
            sleep(Duration::from_secs(2));
            return vec![Default::default()];
        }
        let start = std::time::Instant::now();
        let proofs = match &self.g_stage {
            Some((circuit, public, used_numerics)) => {
                let numeric_config = self.configure(used_numerics.clone());
                let (_, proof) = run_prove_kzg(
                    self.gradient_k,
                    Some(self.dir_path.join("gradient")),
                    circuit.clone(),
                    public.clone(),
                    numeric_config.assigned_num_cols,
                    vec![],
                );
                vec![proof]
            }
            None => panic!("Gradient circuit is not initialized"),
        };
        info!("Prove Gradient Time: {:?}", start.elapsed());
        proofs
    }

    pub fn prove_backward(&self) -> Vec<Proof> {
        if unsafe { NO_PROVE } {
            sleep(Duration::from_secs(10));
            return vec![Default::default()];
        }
        let start = std::time::Instant::now();
        let proofs = match &self.b_stage {
            Some(b_stages) => b_stages
                .par_iter()
                // .iter()
                .enumerate()
                .map(|(i, (circuit, public, used_numerics))| {
                    let numeric_config = self.configure(used_numerics.clone());
                    let (_, proof) = run_prove_kzg(
                        self.backward_k,
                        Some(self.dir_path.join("backward").join(i.to_string())),
                        circuit.clone(),
                        public.clone(),
                        numeric_config.assigned_num_cols,
                        circuit.commitment_tuples(),
                    );
                    proof
                })
                .collect::<Vec<_>>(),
            None => panic!("Backward circuit is not initialized"),
        };
        info!("Prove Backward Time: {:?}", start.elapsed());
        proofs
    }

    pub fn prove(&self) -> Vec<Proof> {
        let mut proofs = vec![];
        proofs.extend(self.prove_forward());
        proofs.extend(self.prove_gradient());
        proofs.extend(self.prove_backward());

        proofs
    }
}

/* Run forward non-circuit and mock */
pub fn run_forward<F: PrimeField + Ord + FromUniformBytes<64>>(
    graph: Graph,
    k: u32,
    mock: bool,
) -> (ForwardCircuit<F>, FieldTensor<F>) {
    configure_static(NumericConfig {
        k,
        ..get_numeric_config()
    });
    // Construct forward circuit
    let mut circuit = ForwardCircuit::<F>::construct(graph.clone());
    // Run forward circuit non-circuit
    let scores = circuit.run().unwrap();
    debug!("scores: {}", scores);
    let scores = scores.mapv(|x| to_field(x));

    // MockProver run
    if mock {
        let public = scores.clone().into_iter().collect();
        debug!("forward_public: {:?}", public);
        let prover = MockProver::run(k, &circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
        info!("Mock Forward Circuit Verified");
    }

    (circuit, scores)
}

/* Run gradient non-circuit and mock */
pub fn run_gradient<F: PrimeField + Ord + FromUniformBytes<64>>(
    scores: FieldTensor<F>,
    labels: Vec<Int>,
    loss: LossType,
    k: u32,
    mock: bool,
) -> (GradientCircuit<F>, FieldTensor<F>) {
    configure_static(NumericConfig {
        k,
        ..get_numeric_config()
    });
    let scores = scores.mapv(|x| to_primitive(&x));

    // Construct gradient circuit
    let circuit = GradientCircuit::<F>::construct(scores.clone(), labels.clone(), loss);
    // Run gradient circuit non-circuit
    let (loss, gradient) = circuit.run().unwrap();
    info!("loss: {}, gradient: {}", loss, gradient);
    let gradient = gradient.mapv(|x| to_field(x));

    // MockProver run
    if mock {
        let public = gradient.clone().into_iter().collect();
        let prover = MockProver::run(k, &circuit, vec![public]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
        info!("Mock Gradient Circuit Verified");
    }

    (circuit, gradient)
}

/* Run backward non-circuit and mock */
pub fn run_backward<F: PrimeField + Ord + FromUniformBytes<64>>(
    gradient: FieldTensor<F>,
    graph: Graph,
    k: u32,
    mock: bool,
) -> (BackwardCircuit<F>, FieldTensor<F>) {
    configure_static(NumericConfig {
        k,
        ..get_numeric_config()
    });

    let gradient = gradient.mapv(|x| to_primitive(&x));
    let mut graph = graph.clone();
    graph.tensor_map.insert("gradient".to_string(), gradient);

    // Construct backward circuit
    let mut circuit = BackwardCircuit::<F>::construct(graph.clone());
    // Run backward circuit non-circuit
    let gradient = circuit.run().unwrap();
    debug!("backward_gradient: {}", gradient);
    let gradient = gradient.mapv(|x| to_field(x));

    // MockProver run
    if mock {
        let public = gradient.clone().into_iter().collect();
        let backward_prover = MockProver::run(k, &circuit, vec![public]).unwrap();
        assert_eq!(backward_prover.verify(), Ok(()));
        info!("Mock Backward Circuit Verified");
    }

    (circuit, gradient)
}
