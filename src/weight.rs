use std::collections::HashMap;

use halo2_proofs::halo2curves::ff::PrimeField;

use crate::{
    graph::Node,
    operations::operation::OPType,
    utils::{
        helpers::{to_field, AssignedTensor, CellRc, FieldTensor, Tensor},
        matcher::match_op_type,
        math::Int,
    },
};

pub struct Weight {
    pub operations: Vec<String>,
    pub weights: Vec<Tensor>,
}

impl Weight {
    pub fn construct(nodes: Vec<Node>, tensor_map: HashMap<String, Tensor>) -> Self {
        let mut operations = Vec::new();
        let mut weights = Vec::new();

        for node in &nodes {
            match match_op_type(node.op_type.clone()) {
                OPType::GEMM => {
                    let weight = tensor_map.get(&node.inputs[1]).unwrap();
                    let weight = weight.clone();
                    operations.push(node.op_type.clone());
                    weights.push(weight);
                }
                _ => {}
            }
        }
        Self {
            operations,
            weights,
        }
    }

    pub fn to_vec(&self) -> Vec<Int> {
        self.weights
            .clone()
            .into_iter()
            .map(|x| x.into_iter())
            .flatten()
            .collect()
    }
}

pub struct FieldWeight<F: PrimeField> {
    pub operations: Vec<String>,
    pub weights: Vec<FieldTensor<F>>,
}

impl<F: PrimeField> FieldWeight<F> {
    pub fn construct(nodes: Vec<Node>, tensor_map: HashMap<String, FieldTensor<F>>) -> Self {
        let mut operations = Vec::new();
        let mut weights = Vec::new();

        for node in &nodes {
            match match_op_type(node.op_type.clone()) {
                OPType::GEMM => {
                    let weight = tensor_map.get(&node.inputs[1]).unwrap();
                    operations.push(node.op_type.clone());
                    weights.push(weight.clone());
                }
                _ => {}
            }
        }
        Self {
            operations,
            weights,
        }
    }

    pub fn construct_from_weight(weight: Weight) -> Self {
        let mut weights = Vec::new();
        for weight in weight.weights {
            let weight = weight.clone();
            weights.push(weight.mapv(|x| to_field::<F>(x)));
        }
        Self {
            operations: weight.operations,
            weights,
        }
    }

    pub fn to_vec(&self) -> Vec<F> {
        self.weights
            .clone()
            .into_iter()
            .map(|x| x.into_iter())
            .flatten()
            .collect()
    }
}

pub struct AssignedWeight<F: PrimeField> {
    pub operations: Vec<String>,
    pub weights: Vec<AssignedTensor<F>>,
}

impl<F: PrimeField> AssignedWeight<F> {
    pub fn construct(nodes: Vec<Node>, tensor_map: HashMap<String, AssignedTensor<F>>) -> Self {
        let mut operations = Vec::new();
        let mut weights = Vec::new();

        for node in &nodes {
            match match_op_type(node.op_type.clone()) {
                OPType::GEMM => {
                    let weight = tensor_map.get(&node.inputs[1]).unwrap();
                    operations.push(node.op_type.clone());
                    weights.push(weight.clone());
                }
                _ => {}
            }
        }
        Self {
            operations,
            weights,
        }
    }

    pub fn to_vec(&self) -> Vec<CellRc<F>> {
        self.weights
            .clone()
            .into_iter()
            .map(|x| x.into_iter())
            .flatten()
            .collect()
    }
}
