use std::{
    fs::create_dir_all,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use halo2_proofs::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{verify_proof, VerifyingKey},
    poly::{
        commitment::ParamsProver,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsVerifierKZG},
            multiopen::VerifierGWC,
            strategy::SingleStrategy,
        },
    },
    transcript::{Blake2bRead, Challenge255, TranscriptReadBuffer},
};
use log::{debug, info};

use crate::{
    prover::helper::{create_params, load_params, save_params},
    utils::helpers::FieldTensor,
};

use super::proof::{Commitment, Proof};

type F = Fr;
type Scheme = KZGCommitmentScheme<Bn256>;

/* Run verify KZG */
pub fn run_verify_kzg(
    vk: &VerifyingKey<G1Affine>,
    public: FieldTensor<F>,
    commitment: Commitment,
    proof: Proof,
) -> bool {
    let mut verifier = Verifier::construct(proof.degree, vk, None);
    verifier.load();
    let public = public.into_iter().collect::<Vec<_>>();
    verifier.verify(proof, vec![public], commitment)
}

pub struct Verifier<'a> {
    pub degree: u32,
    pub dir_path: PathBuf,
    pub verifier_params: Option<ParamsVerifierKZG<Bn256>>,
    pub vk: &'a VerifyingKey<G1Affine>,
}

impl<'a> Verifier<'a> {
    pub fn construct(
        degree: u32,
        vk: &'a VerifyingKey<G1Affine>,
        dir_path: Option<String>,
    ) -> Self {
        Self {
            degree,
            dir_path: match dir_path {
                Some(path) => PathBuf::from_str(&path).unwrap(),
                None => PathBuf::from_str("./params").unwrap(),
            },
            verifier_params: None,
            vk,
        }
    }

    pub fn load(&mut self) {
        self.ensure_dir_exists();
        debug!("Setting params start");
        match self.verifier_params {
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
                self.verifier_params = Some(params.verifier_params().clone());
            }
        }
    }

    pub fn verify(&self, proof: Proof, instances: Vec<Vec<F>>, commitment: Commitment) -> bool {
        let verifier_params = self.verifier_params.as_ref().unwrap();
        let strategy = SingleStrategy::new(&verifier_params);
        let mut verifier_transcript =
            Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof.data[..]);

        let kzg = verify_proof::<Scheme, VerifierGWC<_>, _, _, _>(
            &verifier_params,
            self.vk,
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

    fn ensure_dir_exists(&self) {
        create_dir_all(self.dir_path.clone()).unwrap();
    }
}
