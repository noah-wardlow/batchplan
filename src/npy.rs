//! Minimal NumPy `.npy` writer (format version 1.0, little-endian, C order), so generated
//! datasets load directly with `numpy.load` in training code.

use std::io::Write;
use std::path::Path;

use anyhow::{Result, ensure};

pub trait NpyElement: bytemuck::Pod {
    const DESCR: &'static str;
}

impl NpyElement for f32 {
    const DESCR: &'static str = "<f4";
}

impl NpyElement for i32 {
    const DESCR: &'static str = "<i4";
}

impl NpyElement for u8 {
    const DESCR: &'static str = "|u1";
}

pub fn write_npy<T: NpyElement>(path: impl AsRef<Path>, shape: &[usize], data: &[T]) -> Result<()> {
    ensure!(shape.iter().product::<usize>() == data.len(), "shape {shape:?} does not match {} elements", data.len());
    let dims = match shape {
        [n] => format!("{n},"),
        _ => shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({dims}), }}", T::DESCR);
    // Magic (6) + version (2) + header length (2) + header must be a multiple of 64.
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(bytemuck::cast_slice(data))?;
    f.flush()?;
    Ok(())
}
