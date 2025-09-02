use halo2_proofs::halo2curves::bn256::Fr;
use tonic::{transport::Channel, Status};

use crate::{prover::proof::Proof, utils::helpers::FieldTensor};

use super::server::federated::{
    federated_service_client::FederatedServiceClient, DataRequest, ProofRequest,
};

type F = Fr;

pub struct FederatedClient {
    pub client: FederatedServiceClient<Channel>,
}

impl FederatedClient {
    pub async fn connect(host: String) -> Self {
        Self {
            client: FederatedServiceClient::connect(host).await.unwrap(),
        }
    }

    pub async fn send_data(
        &mut self,
        data: (String, FieldTensor<F>),
    ) -> Result<Vec<Proof>, Status> {
        let data_bytes = bincode::serialize(&data).unwrap();
        let request = tonic::Request::new(DataRequest { data: data_bytes });
        let response = self.client.send_data(request).await.unwrap();
        let proofs_bytes = response.into_inner().proofs;
        let proofs: Vec<Proof> = bincode::deserialize(&proofs_bytes).unwrap();
        Ok(proofs)
    }

    pub async fn send_proof(
        &mut self,
        proofs: (String, Vec<Proof>),
    ) -> Result<FieldTensor<F>, Status> {
        let proofs_bytes = bincode::serialize(&proofs).unwrap();
        let request = tonic::Request::new(ProofRequest {
            proofs: proofs_bytes,
        });
        let response = self.client.send_proof(request).await.unwrap();
        let data_bytes = response.into_inner().data;
        let data: FieldTensor<F> = bincode::deserialize(&data_bytes).unwrap();
        Ok(data)
    }
}
