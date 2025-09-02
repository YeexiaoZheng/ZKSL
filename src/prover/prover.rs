use crate::{
    numeric::NumericConfig,
    prover::{
        helper::{create_keys, create_params, load_params, load_pk, save_params, save_pk, save_vk},
        proof::{Commitment, Proof},
    },
    utils::helpers::{configure_static, get_numeric_config, FieldTensor},
};

use halo2_proofs::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_wrapped_proof, verify_proof, Circuit, ProvingKey},
    poly::{
        commitment::ParamsProver,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::SingleStrategy,
        },
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use log::{debug, info};
use rand::rngs::OsRng;

use std::{
    fmt::Debug,
    fs::create_dir_all,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

type F = Fr;
type Scheme = KZGCommitmentScheme<Bn256>;

/* Run prove KZG */
pub fn run_prove_kzg<ConcreteCircuit: Circuit<F> + Clone + Debug>(
    k: u32,
    dir_path: Option<PathBuf>,
    circuit: ConcreteCircuit,
    public: FieldTensor<F>,
    assigned_num_cols: usize,
    commitment_tuples: Vec<(usize, usize)>,
) -> (ProvingKey<G1Affine>, Proof) {
    configure_static(NumericConfig {
        k,
        ..get_numeric_config()
    });
    let mut prover = ProverKZG::construct(k, circuit.clone(), dir_path);
    prover.load();
    let public = public.into_iter().collect::<Vec<_>>();
    let proof = prover.prove(vec![public.clone()], assigned_num_cols, commitment_tuples);
    prover.verify(proof.clone(), vec![public], proof.commitment.clone());
    (prover.pk.unwrap(), proof)
}

#[derive(Clone)]
pub struct ProverKZG<ConcreteCircuit: Circuit<F> + Clone + Debug> {
    circuit: ConcreteCircuit,
    pub degree: u32,
    pub dir_path: PathBuf,
    pub params: Option<ParamsKZG<Bn256>>,
    pub pk: Option<ProvingKey<G1Affine>>,
}

impl<ConcreteCircuit: Circuit<F> + Clone + Debug> ProverKZG<ConcreteCircuit> {
    pub fn construct(degree: u32, circuit: ConcreteCircuit, dir_path: Option<PathBuf>) -> Self {
        Self {
            circuit,
            degree,
            dir_path: match dir_path {
                Some(path) => path,
                None => PathBuf::from_str("./params").unwrap(),
            },
            params: None,
            pk: None,
        }
    }

    pub fn load(&mut self) {
        self.set_params();
        self.set_pk();
    }

    pub fn prove(
        &self,
        instances: Vec<Vec<F>>,
        assigned_column_size: usize,
        commitment_tuples: Vec<(usize, usize)>,
    ) -> Proof {
        info!("Proving ...");
        // let instances = self.circuit.instances();
        let instances_refs_intermediate = instances.iter().map(|v| &v[..]).collect::<Vec<&[Fr]>>();
        // let instances_refs_intermediate = instances.clone();
        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
        let now = Instant::now();
        let commitments = create_wrapped_proof::<Scheme, ProverGWC<'_, Bn256>, _, _, _, _>(
            self.params.as_ref().unwrap(),
            self.pk.as_ref().unwrap(),
            &[self.circuit.clone()],
            &[&instances_refs_intermediate],
            OsRng,
            &mut transcript,
            assigned_column_size,
            commitment_tuples,
        )
        .unwrap();
        let elapsed = now.elapsed();
        info!("Proof generation took {:?}", elapsed);

        let commitment = Commitment::from_vec(commitments);
        debug!("Proof commitment: {:?}", commitment);

        let proof = transcript.finalize();
        Proof::from("test circuit".to_string(), self.degree, proof, commitment)
    }

    pub fn verify(&self, proof: Proof, instances: Vec<Vec<F>>, commitment: Commitment) -> bool {
        let verifier_params = self.params.as_ref().unwrap().verifier_params();
        let strategy = SingleStrategy::new(&verifier_params);
        // let instance_refs_intermediate = proof
        //     .instances()
        //     .iter()
        //     .map(|v| &v[..])
        //     .collect::<Vec<_>>();
        let mut verifier_transcript =
            Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof.data[..]);

        let kzg = verify_proof::<
            Scheme,
            VerifierGWC<'_, Bn256>,
            Challenge255<G1Affine>,
            Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
            SingleStrategy<'_, Bn256>,
        >(
            &verifier_params,
            &self.pk.as_ref().unwrap().get_vk(),
            strategy,
            &[&instances.iter().map(|x| x.as_slice()).collect::<Vec<_>>()],
            &mut verifier_transcript,
            proof.degree as u64,
        )
        .is_ok();

        let cmt = proof.commitment == commitment;

        debug!("KZG: {}, CMT: {}", kzg, cmt);

        kzg && cmt
    }

    pub fn verifier(&self) -> () {
        // RealVerifier {
        //     circuit_name: derive_circuit_name(&self.circuit),
        //     dir_path: self.dir_path.clone(),
        //     num_instance: self.circuit.num_instance(),
        //     general_params: self
        //         .general_params
        //         .clone()
        //         .ok_or("params not available, please execute prover.load() first")
        //         .unwrap(),
        //     verifier_params: self.params.clone().unwrap(),
        //     circuit_verifying_key: self.circuit_verifying_key.clone().unwrap(),
        // }
    }

    pub fn degree(mut self, k: u32) -> Self {
        self.degree = k;
        self
    }

    fn set_params(&mut self) {
        self.ensure_dir_exists();
        debug!("Setting params start");
        match self.params {
            Some(_) => {
                debug!("Already set params");
            }
            None => {
                let start = Instant::now();
                let path = self
                    .dir_path
                    .join(Path::new(&format!("kzg_{}.params", self.degree)));
                let params = match load_params::<Scheme>(path.clone()) {
                    Ok(params) => params,
                    Err(_) => {
                        let params = create_params::<Scheme>(self.degree).unwrap();
                        save_params::<Scheme>(&path, &params).unwrap();
                        params
                    }
                };

                info!("Setting params took {:?}", start.elapsed());
                self.params = Some(params);
            }
        }
    }

    pub fn set_pk(&mut self) {
        self.ensure_dir_exists();
        debug!("Setting pk start");
        match self.pk {
            Some(_) => {
                debug!("Already set pk");
            }
            None => {
                let start = Instant::now();
                let params = self.params.as_ref().unwrap();
                let path = self
                    .dir_path
                    .join(Path::new(&format!("kzg_{}.pk", self.degree)));
                let pk = match load_pk::<Scheme, ConcreteCircuit>(path.clone()) {
                    Ok(pk) => pk,
                    Err(_) => {
                        let pk =
                            create_keys::<Scheme, ConcreteCircuit>(&self.circuit, params).unwrap();
                        save_pk(&path, &pk).unwrap();
                        let path = self
                            .dir_path
                            .join(Path::new(&format!("kzg_{}.vk", self.degree)));
                        save_vk(&path, &pk.get_vk()).unwrap();
                        pk
                    }
                };
                // let pk = create_keys::<Scheme, ConcreteCircuit>(&self.circuit, params).unwrap();
                info!("Setting pk took {:?}", start.elapsed());
                self.pk = Some(pk);
            }
        }
    }

    fn ensure_dir_exists(&self) {
        create_dir_all(self.dir_path.clone()).unwrap();
    }
}
