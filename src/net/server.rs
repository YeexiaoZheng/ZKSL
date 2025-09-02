use bincode;
use federated::federated_service_server::FederatedService;
use federated::{DataRequest, DataResponse, ProofRequest, ProofResponse};
use halo2_proofs::halo2curves::bn256::Fr;
use log::{debug, info};
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use tonic::{Request, Response, Status};

use crate::graph::Graph;

use crate::loss::LossType;
use crate::net::handler::handle;
use crate::prover::proof::Proof;
use crate::utils::helpers::FieldTensor;

pub mod federated {
    tonic::include_proto!("federated");
}

type F = Fr;

#[derive(Default)]
pub struct FederatedServer {
    pub ks: [u32; 3],
    pub graph: Arc<Mutex<Graph>>,
    pub loss: LossType,
    pub c_num: usize,
    pub r_data: Arc<Mutex<BTreeMap<String, FieldTensor<F>>>>,
    pub s_data: Arc<(Mutex<BTreeMap<String, FieldTensor<F>>>, Condvar)>,
    pub s_proofs: Arc<(Mutex<BTreeMap<String, Vec<Proof>>>, Condvar)>,
    pub round: usize,
    pub parallel: bool,
    pub path: String,
}

#[tonic::async_trait]
impl FederatedService for FederatedServer {
    async fn send_data(
        &self,
        request: Request<DataRequest>,
    ) -> Result<Response<ProofResponse>, Status> {
        let start = std::time::Instant::now();
        let data_bytes = request.into_inner().data;
        let r_data: (String, FieldTensor<F>) = bincode::deserialize(&data_bytes).unwrap();
        debug!("Received data: {:?}", r_data);

        // Store received data
        let lock = &*self.r_data;
        let mut data = lock.lock().unwrap();
        data.insert(r_data.0.clone(), r_data.1);

        // Start handle data
        if data.len() == self.c_num {
            info!("Starting handle_cnn_mlp");
            let r_data = data.clone();
            let ks = self.ks.clone();
            let loss = self.loss.clone();
            let graph = self.graph.clone();
            let s_data = self.s_data.clone();
            let s_proofs = self.s_proofs.clone();
            let parallel = self.parallel.clone();
            let path = self.path.clone();
            thread::spawn(move || {
                handle(r_data, ks, loss, graph, s_data, s_proofs, parallel, path);
            });
        }

        // Wait for proofs to be ready
        let (lock, cvar) = &*self.s_proofs;
        let mut proofs = lock.lock().unwrap();
        while !proofs.contains_key(&r_data.0) {
            proofs = cvar.wait(proofs).unwrap();
        }

        // Return stored proof
        let proof_bytes = bincode::serialize(&proofs.clone().get(&r_data.0).unwrap()).unwrap();

        // Clear stored proofs
        proofs.remove(&r_data.0);

        info!("send_data process time: {:?}", start.elapsed());

        Ok(Response::new(ProofResponse {
            proofs: proof_bytes,
        }))
    }

    async fn send_proof(
        &self,
        request: Request<ProofRequest>,
    ) -> Result<Response<DataResponse>, Status> {
        let start = std::time::Instant::now();
        let proof_bytes = request.into_inner().proofs;
        let proofs: (String, Vec<Proof>) = bincode::deserialize(&proof_bytes).unwrap();
        debug!("Received proof: {:?}", proofs);

        // Verify proofs

        // Wait for data to be ready
        let (lock, cvar) = &*self.s_data;
        let mut data = lock.lock().unwrap();
        while !data.contains_key(&proofs.0) {
            data = cvar.wait(data).unwrap();
        }

        // Return stored data
        let data_bytes = bincode::serialize(&data.get(&proofs.0).unwrap()).unwrap();

        // Clear stored data
        data.remove(&proofs.0);

        info!("send_proof process time: {:?}", start.elapsed());

        Ok(Response::new(DataResponse { data: data_bytes }))
    }
}
