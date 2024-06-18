use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};
use zkml::{graph::Graph, utils::loader::load_from_json};

// use zkml::{graph::Graph, model::ModelCircuit, utils::loader::load_from_json};

fn main() {
    // load graph
    let graph = Graph::construct(load_from_json("src/utils/test.json"));
    println!("{:?}", graph);

    // let circuit = ModelCircuit::<Fr>::construct(3, graph);

    // MockProver::run(3, &circuit, vec![]).unwrap();
}
