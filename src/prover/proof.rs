use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum CommitmentType {
    #[default]
    None,
    Params,
    Input,
    InputOutput,
    ParamsInput,
    ParamsInputOutput,
    UpdatedParamsInput,
    UpdatedParamsInputOutput,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Commitment {
    pub io: (Option<[u8; 32]>, Option<[u8; 32]>),
    pub params: (Option<[u8; 32]>, Option<[u8; 32]>),
}

impl Commitment {
    pub fn from(
        io: (Option<[u8; 32]>, Option<[u8; 32]>),
        params: (Option<[u8; 32]>, Option<[u8; 32]>),
    ) -> Self {
        Self { io, params }
    }

    pub fn from_vec(commitments: Vec<[u8; 32]>) -> Self {
        let (io, params) = match commitments.len() {
            0 => ((None, None), (None, None)),
            1 => ((None, None), (Some(commitments[0]), None)),
            2 => ((None, None), (Some(commitments[0]), Some(commitments[1]))),
            3 => (
                (Some(commitments[0]), Some(commitments[2])),
                (Some(commitments[1]), None),
            ),
            4 => (
                (Some(commitments[0]), Some(commitments[2])),
                (Some(commitments[1]), Some(commitments[3])),
            ),
            _ => panic!("Invalid commitment length"),
        };

        Self { io, params }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Proof {
    pub circuit_name: String,
    pub degree: u32,
    pub data: Vec<u8>,
    pub commitment: Commitment,
}

impl Proof {
    pub fn from(circuit_name: String, degree: u32, proof: Vec<u8>, commitment: Commitment) -> Self {
        Self {
            circuit_name,
            degree,
            data: proof,
            commitment,
        }
    }

    pub fn read_from_file(path: &PathBuf) -> Self {
        let mut file = File::open(path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        serde_json::from_str(&contents).unwrap()
    }

    pub fn write_to_file(&self, path: &PathBuf) {
        let mut file = File::create(path).unwrap();
        file.write_all(serde_json::to_string(self).unwrap().as_bytes())
            .unwrap();
    }
}
