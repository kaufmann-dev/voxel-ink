use std::io::{Read, Write};

use crate::{
    Result,
    cancel::CancellationToken,
    volume::{VolumeGeometry, VolumeView},
};

pub mod binvox;
pub mod nifti;
pub mod vox;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Endian {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NiftiVersion {
    One,
    Two,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Compression {
    None,
    Gzip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FormatFamily {
    Nifti,
    Binvox,
    Vox,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceVariant {
    Nifti {
        version: NiftiVersion,
        compression: Compression,
    },
    Binvox {
        version: u32,
    },
    Vox,
}

impl SourceVariant {
    #[must_use]
    pub const fn family(self) -> FormatFamily {
        match self {
            Self::Nifti { .. } => FormatFamily::Nifti,
            Self::Binvox { .. } => FormatFamily::Binvox,
            Self::Vox => FormatFamily::Vox,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NiftiExtension {
    pub code: i32,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NiftiMetadata {
    pub version: NiftiVersion,
    pub endian: Endian,
    pub dim_info: u8,
    pub intent_parameters: [f64; 3],
    pub intent_code: i32,
    pub pixdim: [f64; 8],
    pub slice_start: i64,
    pub slice_end: i64,
    pub slice_code: i32,
    pub xyzt_units: u8,
    pub slice_duration: f64,
    pub time_offset: f64,
    pub description: Vec<u8>,
    pub auxiliary_file: Vec<u8>,
    pub qform_code: i32,
    pub sform_code: i32,
    pub quaternion: [f64; 3],
    pub qoffset: [f64; 3],
    pub srow: [[f64; 4]; 3],
    pub intent_name: Vec<u8>,
    pub extensions: Vec<NiftiExtension>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinvoxMetadata {
    pub version: u32,
    pub dimensions: [u32; 3],
    pub translation: [f64; 3],
    pub scale: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VoxMetadata {
    pub dimensions: [u32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub enum CodecSaveState {
    Nifti(Box<NiftiMetadata>),
    Binvox(BinvoxMetadata),
    Vox(VoxMetadata),
}

impl CodecSaveState {
    #[must_use]
    pub const fn family(&self) -> FormatFamily {
        match self {
            Self::Nifti(_) => FormatFamily::Nifti,
            Self::Binvox(_) => FormatFamily::Binvox,
            Self::Vox(_) => FormatFamily::Vox,
        }
    }
}

#[derive(Debug)]
pub struct DecodedVolume {
    pub occupancy: Vec<u8>,
    pub geometry: VolumeGeometry,
    pub save_state: CodecSaveState,
    pub source_variant: SourceVariant,
    pub normalized_non_binary: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    pub source_variant: SourceVariant,
}

#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    pub destination_variant: SourceVariant,
}

pub trait VolumeCodec: Send + Sync {
    fn family(&self) -> FormatFamily;

    fn decode(
        &self,
        reader: &mut dyn Read,
        options: DecodeOptions,
        cancel: &CancellationToken,
    ) -> Result<DecodedVolume>;

    fn encode(
        &self,
        volume: VolumeView<'_>,
        state: &CodecSaveState,
        writer: &mut dyn Write,
        options: EncodeOptions,
        cancel: &CancellationToken,
    ) -> Result<()>;
}

#[must_use]
pub fn codec_for_family(family: FormatFamily) -> &'static dyn VolumeCodec {
    match family {
        FormatFamily::Nifti => &nifti::NIFTI_CODEC,
        FormatFamily::Binvox => &binvox::BINVOX_CODEC,
        FormatFamily::Vox => &vox::VOX_CODEC,
    }
}
