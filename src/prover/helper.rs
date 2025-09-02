use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use halo2_proofs::{
    arithmetic::CurveAffine,
    halo2curves::{
        ff::{FromUniformBytes, PrimeField},
        serde::SerdeObject,
    },
    plonk::{keygen_pk, keygen_vk, Circuit, ProvingKey, VerifyingKey},
    poly::commitment::{CommitmentScheme, Params},
};
use log::trace;

use super::error::Error;

// #[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
const KEY_FORMAT: &str = "raw-bytes";

// #[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
const BUF_CAPACITY: &usize = &8000;

/// Creates [CommitmentScheme] parameters for a specific `k` value.
pub fn create_params<Scheme: CommitmentScheme>(k: u32) -> Result<Scheme::ParamsProver, Error> {
    trace!("creating parameters for k={}", k);
    let now = Instant::now();
    let params = Scheme::new_params(k);
    let elapsed = now.elapsed();
    trace!(
        "params took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    Ok(params)
}

/// Saves [CommitmentScheme] parameters to `path`.
pub fn save_params<Scheme: CommitmentScheme>(
    path: &PathBuf,
    params: &'_ Scheme::ParamsProver,
) -> Result<(), Error> {
    trace!("saving parameters 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*BUF_CAPACITY, f);
    params.write(&mut writer)?;
    writer.flush()?;
    Ok(())
}

/// Loads [CommitmentScheme] parameters from `path`.
pub fn load_params<Scheme: CommitmentScheme>(path: PathBuf) -> Result<Scheme::ParamsProver, Error> {
    trace!("loading parameters from {:?}", path);
    let f = File::open(path.clone())?;
    let mut reader = BufReader::with_capacity(*BUF_CAPACITY, f);
    let params = Scheme::ParamsProver::read(&mut reader)?;
    trace!("loaded parameters ✅");
    Ok(params)
}

/// Creates a [VerifyingKey] and [ProvingKey] for a [crate::graph::GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`).
pub fn create_keys<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    circuit: &C,
    params: &'_ Scheme::ParamsProver,
) -> Result<ProvingKey<Scheme::Curve>, Error>
where
    C: Circuit<Scheme::Scalar>,
    <Scheme as CommitmentScheme>::Scalar: FromUniformBytes<64>,
{
    //	Real proof
    let _empty_circuit = <C as Circuit<Scheme::Scalar>>::without_witnesses(circuit);

    // Initialize verifying key
    let now = Instant::now();
    trace!("preparing VK");
    let vk = keygen_vk(params, circuit)?;
    let elapsed = now.elapsed();
    trace!("VK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());

    // Initialize the proving key
    let now = Instant::now();
    let pk = keygen_pk(params, vk, circuit)?;
    let elapsed = now.elapsed();
    trace!("PK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());
    Ok(pk)
}

fn serde_format_from_str(s: &str) -> halo2_proofs::SerdeFormat {
    match s {
        "processed" => halo2_proofs::SerdeFormat::Processed,
        "raw-bytes-unchecked" => halo2_proofs::SerdeFormat::RawBytesUnchecked,
        "raw-bytes" => halo2_proofs::SerdeFormat::RawBytes,
        _ => panic!("invalid serde format"),
    }
}

/// Loads a [VerifyingKey] at `path`.
pub fn load_vk<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    path: PathBuf,
    // cs: ConstraintSystemMid<Scheme::Scalar>,
) -> Result<VerifyingKey<Scheme::Curve>, Error>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    trace!("loading verification key from {:?}", path);
    let f = File::open(path.clone())?;
    let mut reader = BufReader::with_capacity(*BUF_CAPACITY, f);
    let vk = VerifyingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        serde_format_from_str(&KEY_FORMAT),
        // cs.into(),
    )?;
    trace!("loaded verification key ✅");
    Ok(vk)
}

/// Loads a [ProvingKey] at `path`.
pub fn load_pk<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    path: PathBuf,
    // cs: ConstraintSystemMid<Scheme::Scalar>,
) -> Result<ProvingKey<Scheme::Curve>, Error>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    trace!("loading proving key from {:?}", path);
    let f = File::open(path.clone())?;
    let mut reader = BufReader::with_capacity(*BUF_CAPACITY, f);
    let pk = ProvingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        serde_format_from_str(&KEY_FORMAT),
        // cs.into(),
    )?;
    trace!("loaded proving key ✅");
    Ok(pk)
}

/// Saves a [ProvingKey] to `path`.
pub fn save_pk<C: SerdeObject + CurveAffine>(
    path: &PathBuf,
    pk: &ProvingKey<C>,
) -> Result<(), Error>
where
    C::ScalarExt: FromUniformBytes<64> + SerdeObject,
{
    trace!("saving proving key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*BUF_CAPACITY, f);
    pk.write(&mut writer, serde_format_from_str(&KEY_FORMAT))?;
    writer.flush()?;
    trace!("done saving proving key ✅");
    Ok(())
}

/// Saves a [VerifyingKey] to `path`.
pub fn save_vk<C: CurveAffine + SerdeObject>(
    path: &PathBuf,
    vk: &VerifyingKey<C>,
) -> Result<(), Error>
where
    C::ScalarExt: FromUniformBytes<64> + SerdeObject,
{
    trace!("saving verification key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*BUF_CAPACITY, f);
    vk.write(&mut writer, serde_format_from_str(&KEY_FORMAT))?;
    writer.flush()?;
    trace!("done saving verification key ✅");
    Ok(())
}
