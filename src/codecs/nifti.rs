use std::io::{ErrorKind, Read, Write};

use crate::{
    AppError, Result,
    cancel::CancellationToken,
    codecs::{
        CodecSaveState, DecodeOptions, DecodedVolume, EncodeOptions, Endian, FormatFamily,
        NiftiExtension, NiftiMetadata, NiftiVersion, SourceVariant, VolumeCodec,
    },
    error::{MAX_INPUT_BYTES, checked_voxel_count},
    volume::{SpatialUnits, VolumeGeometry, VolumeView},
};

const FORMAT: &str = "NIfTI";
const NIFTI_1_HEADER_SIZE: usize = 348;
const NIFTI_2_HEADER_SIZE: usize = 540;
const EXTENDER_SIZE: usize = 4;
const NIFTI_1_PREFIX_SIZE: usize = NIFTI_1_HEADER_SIZE + EXTENDER_SIZE;
const NIFTI_2_PREFIX_SIZE: usize = NIFTI_2_HEADER_SIZE + EXTENDER_SIZE;
const IO_CHUNK_SIZE: usize = 8 * 1024;

const NIFTI_1_MAGIC: [u8; 4] = *b"n+1\0";
const NIFTI_1_PAIR_MAGIC: [u8; 4] = *b"ni1\0";
const NIFTI_2_MAGIC: [u8; 8] = [b'n', b'+', b'2', 0, 13, 10, 26, 10];
const NIFTI_2_PAIR_MAGIC: [u8; 8] = [b'n', b'i', b'2', 0, 13, 10, 26, 10];

pub static NIFTI_CODEC: NiftiCodec = NiftiCodec;

#[derive(Clone, Copy, Debug, Default)]
pub struct NiftiCodec;

#[derive(Clone, Copy, Debug)]
enum ScalarType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,
    Float32,
    Float64,
}

impl ScalarType {
    fn from_header(datatype: i16, bitpix: i16) -> Result<Self> {
        let (scalar, expected_bitpix) = match datatype {
            2 => (Self::Uint8, 8),
            4 => (Self::Int16, 16),
            8 => (Self::Int32, 32),
            16 => (Self::Float32, 32),
            64 => (Self::Float64, 64),
            256 => (Self::Int8, 8),
            512 => (Self::Uint16, 16),
            768 => (Self::Uint32, 32),
            1024 => (Self::Int64, 64),
            1280 => (Self::Uint64, 64),
            1 => {
                return Err(unsupported(
                    "binary (bit-packed) NIfTI voxels are not supported",
                ));
            }
            1536 => {
                return Err(unsupported("float128 NIfTI voxels are not supported"));
            }
            32 | 1792 | 2048 => {
                return Err(unsupported("complex NIfTI voxels are not supported"));
            }
            128 | 2304 => {
                return Err(unsupported("RGB and RGBA NIfTI voxels are not supported"));
            }
            other => {
                return Err(unsupported(format!(
                    "NIfTI datatype code {other} is not a supported scalar datatype"
                )));
            }
        };
        if bitpix != expected_bitpix {
            return Err(invalid(format!(
                "datatype {datatype} requires bitpix {expected_bitpix}, not {bitpix}"
            )));
        }
        Ok(scalar)
    }

    const fn bytes(self) -> usize {
        match self {
            Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn read_as_f64(self, bytes: &[u8], endian: Endian) -> f64 {
        match self {
            Self::Uint8 => f64::from(bytes[0]),
            Self::Int8 => f64::from(i8::from_ne_bytes([bytes[0]])),
            Self::Uint16 => f64::from(endian.read_u16(bytes, 0)),
            Self::Int16 => f64::from(endian.read_i16(bytes, 0)),
            Self::Uint32 => f64::from(endian.read_u32(bytes, 0)),
            Self::Int32 => f64::from(endian.read_i32(bytes, 0)),
            Self::Uint64 => endian.read_u64(bytes, 0) as f64,
            Self::Int64 => endian.read_i64(bytes, 0) as f64,
            Self::Float32 => f64::from(endian.read_f32(bytes, 0)),
            Self::Float64 => endian.read_f64(bytes, 0),
        }
    }
}

#[derive(Debug)]
struct ParsedHeader {
    dimensions: [u32; 3],
    scalar: ScalarType,
    voxel_offset: u64,
    scale_slope: f64,
    scale_intercept: f64,
    metadata: NiftiMetadata,
}

impl VolumeCodec for NiftiCodec {
    fn family(&self) -> FormatFamily {
        FormatFamily::Nifti
    }

    fn decode(
        &self,
        reader: &mut dyn Read,
        options: DecodeOptions,
        cancel: &CancellationToken,
    ) -> Result<DecodedVolume> {
        let compression = match options.source_variant {
            SourceVariant::Nifti { compression, .. } => compression,
            variant => {
                return Err(unsupported(format!(
                    "the NIfTI codec cannot decode {:?}",
                    variant.family()
                )));
            }
        };

        let mut consumed = 0_u64;
        let mut size_bytes = [0_u8; 4];
        read_exact_limited(
            reader,
            &mut consumed,
            &mut size_bytes,
            "header size",
            cancel,
        )?;
        let (version, endian, header_size) = detect_version_and_endian(size_bytes)?;
        let mut header = allocate_zeroed(header_size)?;
        header[..4].copy_from_slice(&size_bytes);
        read_exact_limited(reader, &mut consumed, &mut header[4..], "header", cancel)?;
        validate_magic(&header, version)?;

        let mut parsed = parse_header(&header, version, endian)?;
        let voxel_count = checked_voxel_count(parsed.dimensions)?;
        let payload_bytes = (voxel_count as u64)
            .checked_mul(parsed.scalar.bytes() as u64)
            .ok_or(AppError::InputLimit)?;
        let payload_end = parsed
            .voxel_offset
            .checked_add(payload_bytes)
            .ok_or(AppError::InputLimit)?;
        if payload_end > MAX_INPUT_BYTES {
            return Err(AppError::InputLimit);
        }

        let mut extender = [0_u8; EXTENDER_SIZE];
        read_exact_limited(
            reader,
            &mut consumed,
            &mut extender,
            "extension indicator",
            cancel,
        )?;
        if extender[1..] != [0, 0, 0] {
            return Err(invalid("the NIfTI extension indicator is malformed"));
        }
        parsed.metadata.extensions = read_extensions(
            reader,
            &mut consumed,
            parsed.voxel_offset,
            extender[0] != 0,
            endian,
            cancel,
        )?;

        let (occupancy, normalized_non_binary) =
            decode_payload(reader, &mut consumed, voxel_count, &parsed, cancel)?;
        drain_to_eof(reader, &mut consumed, cancel)?;

        let geometry = geometry_from_metadata(parsed.dimensions, &parsed.metadata)?;
        Ok(DecodedVolume {
            occupancy,
            geometry,
            save_state: CodecSaveState::Nifti(Box::new(parsed.metadata)),
            source_variant: SourceVariant::Nifti {
                version,
                compression,
            },
            normalized_non_binary,
        })
    }

    fn encode(
        &self,
        volume: VolumeView<'_>,
        state: &CodecSaveState,
        writer: &mut dyn Write,
        options: EncodeOptions,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let CodecSaveState::Nifti(metadata) = state else {
            return Err(invalid("a NIfTI save requires NIfTI source metadata"));
        };
        let destination_version = match options.destination_variant {
            SourceVariant::Nifti { version, .. } => version,
            variant => {
                return Err(AppError::ConversionNotAllowed(format!(
                    "{family:?}",
                    family = variant.family()
                )));
            }
        };
        if destination_version != metadata.version {
            return Err(AppError::ConversionNotAllowed(
                "a different NIfTI header version".into(),
            ));
        }

        let voxel_count = checked_voxel_count(volume.dimensions)?;
        if volume.occupancy.len() != voxel_count {
            return Err(invalid(format!(
                "the occupancy has {} bytes but the dimensions require {voxel_count}",
                volume.occupancy.len()
            )));
        }
        let (header, extension_bytes, padding_bytes) = build_header(volume, metadata, cancel)?;
        let output_bytes = (header.len() as u64)
            .checked_add(extension_bytes)
            .and_then(|bytes| bytes.checked_add(padding_bytes))
            .and_then(|bytes| bytes.checked_add(voxel_count as u64))
            .ok_or(AppError::InputLimit)?;
        if output_bytes > MAX_INPUT_BYTES {
            return Err(AppError::InputLimit);
        }

        write_all_checked(writer, &header, "header", cancel)?;
        write_extensions(writer, &metadata.extensions, metadata.endian, cancel)?;
        write_zero_padding(writer, padding_bytes, cancel)?;
        for chunk in volume.occupancy.chunks(IO_CHUNK_SIZE) {
            if chunk.iter().any(|&value| value > 1) {
                return Err(invalid("occupancy values must be zero or one"));
            }
            write_all_checked(writer, chunk, "voxel payload", cancel)?;
        }
        Ok(())
    }
}

impl Endian {
    fn read_i16(self, bytes: &[u8], offset: usize) -> i16 {
        let value = [bytes[offset], bytes[offset + 1]];
        match self {
            Self::Little => i16::from_le_bytes(value),
            Self::Big => i16::from_be_bytes(value),
        }
    }

    fn read_u16(self, bytes: &[u8], offset: usize) -> u16 {
        let value = [bytes[offset], bytes[offset + 1]];
        match self {
            Self::Little => u16::from_le_bytes(value),
            Self::Big => u16::from_be_bytes(value),
        }
    }

    fn read_i32(self, bytes: &[u8], offset: usize) -> i32 {
        let value = [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ];
        match self {
            Self::Little => i32::from_le_bytes(value),
            Self::Big => i32::from_be_bytes(value),
        }
    }

    fn read_u32(self, bytes: &[u8], offset: usize) -> u32 {
        let value = [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ];
        match self {
            Self::Little => u32::from_le_bytes(value),
            Self::Big => u32::from_be_bytes(value),
        }
    }

    fn read_i64(self, bytes: &[u8], offset: usize) -> i64 {
        let value = [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ];
        match self {
            Self::Little => i64::from_le_bytes(value),
            Self::Big => i64::from_be_bytes(value),
        }
    }

    fn read_u64(self, bytes: &[u8], offset: usize) -> u64 {
        let value = [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ];
        match self {
            Self::Little => u64::from_le_bytes(value),
            Self::Big => u64::from_be_bytes(value),
        }
    }

    fn read_f32(self, bytes: &[u8], offset: usize) -> f32 {
        f32::from_bits(self.read_u32(bytes, offset))
    }

    fn read_f64(self, bytes: &[u8], offset: usize) -> f64 {
        f64::from_bits(self.read_u64(bytes, offset))
    }

    fn write_i16(self, bytes: &mut [u8], offset: usize, value: i16) {
        let encoded = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        bytes[offset..offset + 2].copy_from_slice(&encoded);
    }

    fn write_i32(self, bytes: &mut [u8], offset: usize, value: i32) {
        let encoded = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        bytes[offset..offset + 4].copy_from_slice(&encoded);
    }

    fn write_i64(self, bytes: &mut [u8], offset: usize, value: i64) {
        let encoded = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        bytes[offset..offset + 8].copy_from_slice(&encoded);
    }

    fn write_f32(self, bytes: &mut [u8], offset: usize, value: f32) {
        self.write_u32(bytes, offset, value.to_bits());
    }

    fn write_f64(self, bytes: &mut [u8], offset: usize, value: f64) {
        self.write_u64(bytes, offset, value.to_bits());
    }

    fn write_u32(self, bytes: &mut [u8], offset: usize, value: u32) {
        let encoded = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        bytes[offset..offset + 4].copy_from_slice(&encoded);
    }

    fn write_u64(self, bytes: &mut [u8], offset: usize, value: u64) {
        let encoded = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        bytes[offset..offset + 8].copy_from_slice(&encoded);
    }
}

fn detect_version_and_endian(size_bytes: [u8; 4]) -> Result<(NiftiVersion, Endian, usize)> {
    let little = i32::from_le_bytes(size_bytes);
    let big = i32::from_be_bytes(size_bytes);
    match (little, big) {
        (348, _) => Ok((NiftiVersion::One, Endian::Little, NIFTI_1_HEADER_SIZE)),
        (540, _) => Ok((NiftiVersion::Two, Endian::Little, NIFTI_2_HEADER_SIZE)),
        (_, 348) => Ok((NiftiVersion::One, Endian::Big, NIFTI_1_HEADER_SIZE)),
        (_, 540) => Ok((NiftiVersion::Two, Endian::Big, NIFTI_2_HEADER_SIZE)),
        _ => Err(invalid("the header size is neither NIfTI-1 nor NIfTI-2")),
    }
}

fn validate_magic(header: &[u8], version: NiftiVersion) -> Result<()> {
    let (magic, expected, pair): (&[u8], &[u8], &[u8]) = match version {
        NiftiVersion::One => (&header[344..348], &NIFTI_1_MAGIC, &NIFTI_1_PAIR_MAGIC),
        NiftiVersion::Two => (&header[4..12], &NIFTI_2_MAGIC, &NIFTI_2_PAIR_MAGIC),
    };
    if magic == expected {
        Ok(())
    } else if magic == pair {
        Err(unsupported(
            "detached NIfTI .hdr/.img pairs are not supported",
        ))
    } else {
        Err(invalid("the single-file NIfTI magic is invalid"))
    }
}

fn parse_header(header: &[u8], version: NiftiVersion, endian: Endian) -> Result<ParsedHeader> {
    let dimensions = parse_dimensions(header, version, endian)?;
    let (datatype, bitpix) = match version {
        NiftiVersion::One => (endian.read_i16(header, 70), endian.read_i16(header, 72)),
        NiftiVersion::Two => (endian.read_i16(header, 12), endian.read_i16(header, 14)),
    };
    let scalar = ScalarType::from_header(datatype, bitpix)?;
    let pixdim = read_pixdim(header, version, endian);
    let (scale_slope, scale_intercept) = match version {
        NiftiVersion::One => (
            f64::from(endian.read_f32(header, 112)),
            f64::from(endian.read_f32(header, 116)),
        ),
        NiftiVersion::Two => (endian.read_f64(header, 176), endian.read_f64(header, 184)),
    };
    if !scale_slope.is_finite() {
        return Err(invalid("scl_slope must be finite"));
    }
    if scale_slope != 0.0 && !scale_intercept.is_finite() {
        return Err(invalid(
            "scl_inter must be finite when scl_slope is non-zero",
        ));
    }
    let voxel_offset = parse_voxel_offset(header, version, endian)?;

    let metadata = match version {
        NiftiVersion::One => parse_nifti_1_metadata(header, endian, pixdim),
        NiftiVersion::Two => parse_nifti_2_metadata(header, endian, pixdim)?,
    };
    Ok(ParsedHeader {
        dimensions,
        scalar,
        voxel_offset,
        scale_slope,
        scale_intercept,
        metadata,
    })
}

fn parse_dimensions(header: &[u8], version: NiftiVersion, endian: Endian) -> Result<[u32; 3]> {
    let mut dimensions = [0_i64; 8];
    for (axis, dimension) in dimensions.iter_mut().enumerate() {
        *dimension = match version {
            NiftiVersion::One => i64::from(endian.read_i16(header, 40 + axis * 2)),
            NiftiVersion::Two => endian.read_i64(header, 16 + axis * 8),
        };
    }
    let axis_count = dimensions[0];
    if !(1..=7).contains(&axis_count) {
        return Err(invalid("dim[0] must be between one and seven"));
    }
    let axis_count = usize::try_from(axis_count)
        .map_err(|_| invalid("dim[0] cannot be represented on this platform"))?;
    for (axis, &dimension) in dimensions.iter().enumerate().take(axis_count + 1).skip(1) {
        if dimension <= 0 {
            return Err(invalid(format!("dim[{axis}] must be positive")));
        }
    }
    for (axis, &dimension) in dimensions.iter().enumerate().skip(4) {
        if dimension != 1 {
            return Err(unsupported(format!(
                "dim[{axis}] is {dimension}; only one 3D scalar volume is supported"
            )));
        }
    }

    let mut output = [1_u32; 3];
    for (axis, value) in output.iter_mut().enumerate() {
        if axis < axis_count {
            let axis_number = axis + 1;
            *value = u32::try_from(dimensions[axis_number])
                .map_err(|_| invalid(format!("dim[{axis_number}] is too large")))?;
        }
    }
    checked_voxel_count(output)?;
    Ok(output)
}

fn read_pixdim(header: &[u8], version: NiftiVersion, endian: Endian) -> [f64; 8] {
    let mut pixdim = [0.0; 8];
    for (index, value) in pixdim.iter_mut().enumerate() {
        *value = match version {
            NiftiVersion::One => f64::from(endian.read_f32(header, 76 + index * 4)),
            NiftiVersion::Two => endian.read_f64(header, 104 + index * 8),
        };
    }
    pixdim
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn parse_voxel_offset(header: &[u8], version: NiftiVersion, endian: Endian) -> Result<u64> {
    let minimum = match version {
        NiftiVersion::One => NIFTI_1_PREFIX_SIZE as u64,
        NiftiVersion::Two => NIFTI_2_PREFIX_SIZE as u64,
    };
    let offset = match version {
        NiftiVersion::One => {
            let value = f64::from(endian.read_f32(header, 108));
            if !value.is_finite() {
                return Err(invalid("vox_offset is not finite"));
            }
            if value > 1_200_000_000.0 {
                return Err(AppError::InputLimit);
            }
            let minimum_float = f64::from(
                u32::try_from(minimum).map_err(|_| invalid("invalid minimum vox_offset"))?,
            );
            if value < minimum_float {
                minimum
            } else {
                if value.fract() != 0.0 {
                    return Err(invalid("vox_offset is not an integer"));
                }
                value as u64
            }
        }
        NiftiVersion::Two => {
            let value = endian.read_i64(header, 168);
            let minimum_signed =
                i64::try_from(minimum).map_err(|_| invalid("invalid minimum vox_offset"))?;
            if value < minimum_signed {
                minimum
            } else {
                u64::try_from(value).map_err(|_| invalid("vox_offset is negative"))?
            }
        }
    };
    if offset > MAX_INPUT_BYTES {
        return Err(AppError::InputLimit);
    }
    Ok(offset)
}

fn parse_nifti_1_metadata(header: &[u8], endian: Endian, pixdim: [f64; 8]) -> NiftiMetadata {
    NiftiMetadata {
        version: NiftiVersion::One,
        endian,
        dim_info: header[39],
        intent_parameters: [
            f64::from(endian.read_f32(header, 56)),
            f64::from(endian.read_f32(header, 60)),
            f64::from(endian.read_f32(header, 64)),
        ],
        intent_code: i32::from(endian.read_i16(header, 68)),
        pixdim,
        slice_start: i64::from(endian.read_i16(header, 74)),
        slice_end: i64::from(endian.read_i16(header, 120)),
        slice_code: i32::from(header[122]),
        xyzt_units: header[123],
        slice_duration: f64::from(endian.read_f32(header, 132)),
        time_offset: f64::from(endian.read_f32(header, 136)),
        description: fixed_field(header, 148, 80),
        auxiliary_file: fixed_field(header, 228, 24),
        qform_code: i32::from(endian.read_i16(header, 252)),
        sform_code: i32::from(endian.read_i16(header, 254)),
        quaternion: [
            f64::from(endian.read_f32(header, 256)),
            f64::from(endian.read_f32(header, 260)),
            f64::from(endian.read_f32(header, 264)),
        ],
        qoffset: [
            f64::from(endian.read_f32(header, 268)),
            f64::from(endian.read_f32(header, 272)),
            f64::from(endian.read_f32(header, 276)),
        ],
        srow: read_srow(header, 280, 4, endian),
        intent_name: fixed_field(header, 328, 16),
        extensions: Vec::new(),
    }
}

fn parse_nifti_2_metadata(
    header: &[u8],
    endian: Endian,
    pixdim: [f64; 8],
) -> Result<NiftiMetadata> {
    let units = endian.read_i32(header, 500);
    let xyzt_units = u8::try_from(units).map_err(|_| {
        invalid(format!(
            "xyzt_units value {units} is outside the valid byte range"
        ))
    })?;
    Ok(NiftiMetadata {
        version: NiftiVersion::Two,
        endian,
        dim_info: header[524],
        intent_parameters: [
            endian.read_f64(header, 80),
            endian.read_f64(header, 88),
            endian.read_f64(header, 96),
        ],
        intent_code: endian.read_i32(header, 504),
        pixdim,
        slice_start: endian.read_i64(header, 224),
        slice_end: endian.read_i64(header, 232),
        slice_code: endian.read_i32(header, 496),
        xyzt_units,
        slice_duration: endian.read_f64(header, 208),
        time_offset: endian.read_f64(header, 216),
        description: fixed_field(header, 240, 80),
        auxiliary_file: fixed_field(header, 320, 24),
        qform_code: endian.read_i32(header, 344),
        sform_code: endian.read_i32(header, 348),
        quaternion: [
            endian.read_f64(header, 352),
            endian.read_f64(header, 360),
            endian.read_f64(header, 368),
        ],
        qoffset: [
            endian.read_f64(header, 376),
            endian.read_f64(header, 384),
            endian.read_f64(header, 392),
        ],
        srow: read_srow(header, 400, 8, endian),
        intent_name: fixed_field(header, 508, 16),
        extensions: Vec::new(),
    })
}

fn read_srow(header: &[u8], offset: usize, scalar_size: usize, endian: Endian) -> [[f64; 4]; 3] {
    let mut rows = [[0.0; 4]; 3];
    for (row_index, row) in rows.iter_mut().enumerate() {
        for (column, value) in row.iter_mut().enumerate() {
            let field_offset = offset + (row_index * 4 + column) * scalar_size;
            *value = if scalar_size == 4 {
                f64::from(endian.read_f32(header, field_offset))
            } else {
                endian.read_f64(header, field_offset)
            };
        }
    }
    rows
}

fn fixed_field(header: &[u8], offset: usize, length: usize) -> Vec<u8> {
    let field = &header[offset..offset + length];
    let retained = field
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(0, |index| index + 1);
    field[..retained].to_vec()
}

fn read_extensions(
    reader: &mut dyn Read,
    consumed: &mut u64,
    voxel_offset: u64,
    indicated: bool,
    endian: Endian,
    cancel: &CancellationToken,
) -> Result<Vec<NiftiExtension>> {
    if *consumed > voxel_offset {
        return Err(invalid("vox_offset points inside the header"));
    }
    let mut remaining = voxel_offset - *consumed;
    if !indicated {
        consume_bytes(reader, consumed, remaining, cancel)?;
        return Ok(Vec::new());
    }
    if remaining == 0 {
        return Err(invalid(
            "the extension indicator is set but no extension record follows",
        ));
    }

    let mut extensions = Vec::new();
    while remaining > 0 {
        cancel.check()?;
        if remaining < 8 {
            consume_zero_padding(reader, consumed, remaining, cancel)?;
            break;
        }
        let mut record_header = [0_u8; 8];
        read_exact_limited(
            reader,
            consumed,
            &mut record_header,
            "extension header",
            cancel,
        )?;
        remaining -= 8;
        if record_header == [0; 8] {
            if extensions.is_empty() {
                return Err(invalid(
                    "the extension indicator is set but no valid extension record follows",
                ));
            }
            consume_zero_padding(reader, consumed, remaining, cancel)?;
            break;
        }

        let size = endian.read_i32(&record_header, 0);
        let code = endian.read_i32(&record_header, 4);
        if size < 16 || size % 16 != 0 {
            return Err(invalid(format!(
                "extension size {size} is not a positive multiple of 16"
            )));
        }
        if code < 0 {
            return Err(invalid(format!("extension code {code} is negative")));
        }
        let payload_length = usize::try_from(size - 8)
            .map_err(|_| invalid("extension payload size cannot be represented"))?;
        if payload_length as u64 > remaining {
            return Err(invalid("an extension record extends past vox_offset"));
        }
        let payload = read_vec_limited(
            reader,
            consumed,
            payload_length,
            "extension payload",
            cancel,
        )?;
        remaining -= payload_length as u64;
        extensions
            .try_reserve(1)
            .map_err(|_| AppError::Allocation {
                bytes: (extensions.len() + 1).saturating_mul(size_of::<NiftiExtension>()),
            })?;
        extensions.push(NiftiExtension { code, payload });
    }
    if extensions.is_empty() {
        return Err(invalid(
            "the extension indicator is set but no valid extension record follows",
        ));
    }
    Ok(extensions)
}

fn consume_zero_padding(
    reader: &mut dyn Read,
    consumed: &mut u64,
    mut bytes: u64,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut buffer = [0_u8; IO_CHUNK_SIZE];
    while bytes > 0 {
        let amount =
            usize::try_from(bytes.min(IO_CHUNK_SIZE as u64)).map_err(|_| AppError::InputLimit)?;
        read_exact_limited(
            reader,
            consumed,
            &mut buffer[..amount],
            "padding before the voxel payload",
            cancel,
        )?;
        if buffer[..amount].iter().any(|&byte| byte != 0) {
            return Err(invalid(
                "non-zero bytes appear outside an indicated extension record",
            ));
        }
        bytes -= amount as u64;
    }
    Ok(())
}

fn consume_bytes(
    reader: &mut dyn Read,
    consumed: &mut u64,
    mut bytes: u64,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut buffer = [0_u8; IO_CHUNK_SIZE];
    while bytes > 0 {
        let amount =
            usize::try_from(bytes.min(IO_CHUNK_SIZE as u64)).map_err(|_| AppError::InputLimit)?;
        read_exact_limited(
            reader,
            consumed,
            &mut buffer[..amount],
            "bytes before the voxel payload",
            cancel,
        )?;
        bytes -= amount as u64;
    }
    Ok(())
}

#[allow(clippy::float_cmp, clippy::suboptimal_flops)]
fn decode_payload(
    reader: &mut dyn Read,
    consumed: &mut u64,
    voxel_count: usize,
    header: &ParsedHeader,
    cancel: &CancellationToken,
) -> Result<(Vec<u8>, bool)> {
    cancel.check()?;
    let mut occupancy = Vec::new();
    occupancy
        .try_reserve_exact(voxel_count)
        .map_err(|_| AppError::Allocation { bytes: voxel_count })?;

    let byte_scaling_is_identity =
        header.scale_slope == 0.0 || (header.scale_slope == 1.0 && header.scale_intercept == 0.0);
    if matches!(header.scalar, ScalarType::Uint8 | ScalarType::Int8) && byte_scaling_is_identity {
        occupancy.resize(voxel_count, 0);
        let mut normalized_non_binary = false;
        for chunk in occupancy.chunks_mut(IO_CHUNK_SIZE) {
            read_exact_limited(reader, consumed, chunk, "voxel payload", cancel)?;
            for value in chunk {
                if *value > 1 {
                    normalized_non_binary = true;
                    *value = 1;
                }
            }
        }
        return Ok((occupancy, normalized_non_binary));
    }

    let scalars_per_chunk = (IO_CHUNK_SIZE / header.scalar.bytes()).max(1);
    let mut buffer = [0_u8; IO_CHUNK_SIZE];
    let mut normalized_non_binary = false;
    let mut decoded = 0_usize;
    while decoded < voxel_count {
        cancel.check()?;
        let scalar_count = (voxel_count - decoded).min(scalars_per_chunk);
        let byte_count = scalar_count * header.scalar.bytes();
        read_exact_limited(
            reader,
            consumed,
            &mut buffer[..byte_count],
            "voxel payload",
            cancel,
        )?;
        for local_index in 0..scalar_count {
            let offset = local_index * header.scalar.bytes();
            let raw = header.scalar.read_as_f64(
                &buffer[offset..offset + header.scalar.bytes()],
                header.metadata.endian,
            );
            let effective = if header.scale_slope == 0.0 {
                raw
            } else {
                raw * header.scale_slope + header.scale_intercept
            };
            occupancy.push(u8::from(effective != 0.0));
            normalized_non_binary |= effective != 0.0 && effective != 1.0;
        }
        decoded += scalar_count;
    }
    Ok((occupancy, normalized_non_binary))
}

fn geometry_from_metadata(
    dimensions: [u32; 3],
    metadata: &NiftiMetadata,
) -> Result<VolumeGeometry> {
    let units = match metadata.xyzt_units & 0x07 {
        1 => SpatialUnits::Meter,
        2 => SpatialUnits::Millimeter,
        3 => SpatialUnits::Micron,
        _ => SpatialUnits::Unknown,
    };

    if metadata.sform_code > 0 {
        let affine = rows_to_affine(metadata.srow);
        if let Ok(geometry) = VolumeGeometry::new(dimensions, affine, units) {
            return Ok(geometry);
        }
    }
    if metadata.qform_code > 0
        && let Some(affine) = quaternion_affine(metadata)
        && let Ok(geometry) = VolumeGeometry::new(dimensions, affine, units)
    {
        return Ok(geometry);
    }
    VolumeGeometry::new(dimensions, fallback_affine(metadata.pixdim), units)
}

const fn rows_to_affine(rows: [[f64; 4]; 3]) -> [[f64; 4]; 4] {
    [rows[0], rows[1], rows[2], [0.0, 0.0, 0.0, 1.0]]
}

#[allow(clippy::many_single_char_names, clippy::suboptimal_flops)]
fn quaternion_affine(metadata: &NiftiMetadata) -> Option<[[f64; 4]; 4]> {
    let [mut b, mut c, mut d] = metadata.quaternion;
    if !metadata
        .quaternion
        .into_iter()
        .chain(metadata.qoffset)
        .all(f64::is_finite)
    {
        return None;
    }
    let mut squared_a = 1.0 - (b * b + c * c + d * d);
    let a = if squared_a < 1.0e-7 {
        let length = (b * b + c * c + d * d).sqrt();
        if !length.is_finite() || length <= 0.0 {
            return None;
        }
        b /= length;
        c /= length;
        d /= length;
        squared_a = 0.0;
        0.0
    } else {
        squared_a.sqrt()
    };
    let spacing = [1, 2, 3].map(|axis| valid_spacing(metadata.pixdim[axis]));
    let qfac = if metadata.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
    let [dx, dy, dz] = spacing;
    let [x, y, z] = metadata.qoffset;
    let aa = squared_a;
    let bb = b * b;
    let cc = c * c;
    let dd = d * d;
    Some([
        [
            (aa + bb - cc - dd) * dx,
            2.0 * (b * c - a * d) * dy,
            2.0 * (b * d + a * c) * dz * qfac,
            x,
        ],
        [
            2.0 * (b * c + a * d) * dx,
            (aa + cc - bb - dd) * dy,
            2.0 * (c * d - a * b) * dz * qfac,
            y,
        ],
        [
            2.0 * (b * d - a * c) * dx,
            2.0 * (c * d + a * b) * dy,
            (aa + dd - cc - bb) * dz * qfac,
            z,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn fallback_affine(pixdim: [f64; 8]) -> [[f64; 4]; 4] {
    [
        [valid_spacing(pixdim[1]), 0.0, 0.0, 0.0],
        [0.0, valid_spacing(pixdim[2]), 0.0, 0.0],
        [0.0, 0.0, valid_spacing(pixdim[3]), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn valid_spacing(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    }
}

fn build_header(
    volume: VolumeView<'_>,
    metadata: &NiftiMetadata,
    cancel: &CancellationToken,
) -> Result<(Vec<u8>, u64, u64)> {
    let extension_bytes = extension_byte_count(&metadata.extensions, cancel)?;
    let prefix_size = match metadata.version {
        NiftiVersion::One => NIFTI_1_PREFIX_SIZE,
        NiftiVersion::Two => NIFTI_2_PREFIX_SIZE,
    };
    let minimum_offset = (prefix_size as u64)
        .checked_add(extension_bytes)
        .ok_or(AppError::InputLimit)?;
    let voxel_offset = match metadata.version {
        NiftiVersion::One => representable_nifti_1_offset(minimum_offset),
        NiftiVersion::Two => minimum_offset,
    };
    let padding_bytes = voxel_offset - minimum_offset;
    let mut header = allocate_zeroed(prefix_size)?;
    match metadata.version {
        NiftiVersion::One => {
            write_nifti_1_header(&mut header, volume.dimensions, metadata, voxel_offset)?;
        }
        NiftiVersion::Two => {
            write_nifti_2_header(&mut header, volume.dimensions, metadata, voxel_offset)?;
        }
    }
    header[prefix_size - EXTENDER_SIZE] = u8::from(!metadata.extensions.is_empty());
    Ok((header, extension_bytes, padding_bytes))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn representable_nifti_1_offset(minimum: u64) -> u64 {
    let mut represented = minimum as f32;
    if (represented as u64) < minimum {
        represented = represented.next_up();
    }
    represented as u64
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn write_nifti_1_header(
    header: &mut [u8],
    dimensions: [u32; 3],
    metadata: &NiftiMetadata,
    voxel_offset: u64,
) -> Result<()> {
    let endian = metadata.endian;
    endian.write_i32(header, 0, 348);
    header[39] = metadata.dim_info;
    write_nifti_1_dimensions(header, dimensions, endian)?;
    for (index, &parameter) in metadata.intent_parameters.iter().enumerate() {
        endian.write_f32(
            header,
            56 + index * 4,
            narrow_f32(parameter, "intent parameter")?,
        );
    }
    endian.write_i16(header, 68, narrow_i16(metadata.intent_code, "intent_code")?);
    endian.write_i16(header, 70, 2);
    endian.write_i16(header, 72, 8);
    endian.write_i16(
        header,
        74,
        i16::try_from(metadata.slice_start)
            .map_err(|_| invalid("slice_start does not fit a NIfTI-1 header"))?,
    );
    for (index, &value) in metadata.pixdim.iter().enumerate() {
        endian.write_f32(header, 76 + index * 4, narrow_f32(value, "pixdim")?);
    }
    let offset = voxel_offset as f32;
    if offset as u64 != voxel_offset {
        return Err(invalid(
            "the NIfTI-1 extension area is too large for an exact vox_offset",
        ));
    }
    endian.write_f32(header, 108, offset);
    endian.write_f32(header, 112, 1.0);
    endian.write_f32(header, 116, 0.0);
    endian.write_i16(
        header,
        120,
        i16::try_from(metadata.slice_end)
            .map_err(|_| invalid("slice_end does not fit a NIfTI-1 header"))?,
    );
    header[122] = u8::try_from(metadata.slice_code)
        .map_err(|_| invalid("slice_code does not fit a NIfTI-1 header"))?;
    header[123] = metadata.xyzt_units;
    endian.write_f32(header, 124, 1.0);
    endian.write_f32(header, 128, 0.0);
    endian.write_f32(
        header,
        132,
        narrow_f32(metadata.slice_duration, "slice_duration")?,
    );
    endian.write_f32(
        header,
        136,
        narrow_f32(metadata.time_offset, "time_offset")?,
    );
    endian.write_i32(header, 140, 1);
    endian.write_i32(header, 144, 0);
    write_fixed_field(header, 148, 80, &metadata.description, "description")?;
    write_fixed_field(header, 228, 24, &metadata.auxiliary_file, "auxiliary_file")?;
    endian.write_i16(header, 252, narrow_i16(metadata.qform_code, "qform_code")?);
    endian.write_i16(header, 254, narrow_i16(metadata.sform_code, "sform_code")?);
    for (index, &value) in metadata.quaternion.iter().enumerate() {
        endian.write_f32(header, 256 + index * 4, narrow_f32(value, "quaternion")?);
    }
    for (index, &value) in metadata.qoffset.iter().enumerate() {
        endian.write_f32(header, 268 + index * 4, narrow_f32(value, "qoffset")?);
    }
    write_nifti_1_srow(header, metadata.srow, endian)?;
    write_fixed_field(header, 328, 16, &metadata.intent_name, "intent_name")?;
    header[344..348].copy_from_slice(&NIFTI_1_MAGIC);
    Ok(())
}

fn write_nifti_2_header(
    header: &mut [u8],
    dimensions: [u32; 3],
    metadata: &NiftiMetadata,
    voxel_offset: u64,
) -> Result<()> {
    let endian = metadata.endian;
    endian.write_i32(header, 0, 540);
    header[4..12].copy_from_slice(&NIFTI_2_MAGIC);
    endian.write_i16(header, 12, 2);
    endian.write_i16(header, 14, 8);
    endian.write_i64(header, 16, 3);
    for (axis, &dimension) in dimensions.iter().enumerate() {
        endian.write_i64(header, 24 + axis * 8, i64::from(dimension));
    }
    for axis in 4..8 {
        endian.write_i64(header, 16 + axis * 8, 1);
    }
    for (index, &parameter) in metadata.intent_parameters.iter().enumerate() {
        endian.write_f64(header, 80 + index * 8, parameter);
    }
    for (index, &value) in metadata.pixdim.iter().enumerate() {
        endian.write_f64(header, 104 + index * 8, value);
    }
    endian.write_i64(
        header,
        168,
        i64::try_from(voxel_offset).map_err(|_| AppError::InputLimit)?,
    );
    endian.write_f64(header, 176, 1.0);
    endian.write_f64(header, 184, 0.0);
    endian.write_f64(header, 192, 1.0);
    endian.write_f64(header, 200, 0.0);
    endian.write_f64(header, 208, metadata.slice_duration);
    endian.write_f64(header, 216, metadata.time_offset);
    endian.write_i64(header, 224, metadata.slice_start);
    endian.write_i64(header, 232, metadata.slice_end);
    write_fixed_field(header, 240, 80, &metadata.description, "description")?;
    write_fixed_field(header, 320, 24, &metadata.auxiliary_file, "auxiliary_file")?;
    endian.write_i32(header, 344, metadata.qform_code);
    endian.write_i32(header, 348, metadata.sform_code);
    for (index, &value) in metadata.quaternion.iter().enumerate() {
        endian.write_f64(header, 352 + index * 8, value);
    }
    for (index, &value) in metadata.qoffset.iter().enumerate() {
        endian.write_f64(header, 376 + index * 8, value);
    }
    for (row_index, row) in metadata.srow.iter().enumerate() {
        for (column, &value) in row.iter().enumerate() {
            endian.write_f64(header, 400 + (row_index * 4 + column) * 8, value);
        }
    }
    endian.write_i32(header, 496, metadata.slice_code);
    endian.write_i32(header, 500, i32::from(metadata.xyzt_units));
    endian.write_i32(header, 504, metadata.intent_code);
    write_fixed_field(header, 508, 16, &metadata.intent_name, "intent_name")?;
    header[524] = metadata.dim_info;
    Ok(())
}

fn write_nifti_1_dimensions(header: &mut [u8], dimensions: [u32; 3], endian: Endian) -> Result<()> {
    endian.write_i16(header, 40, 3);
    for (axis, &dimension) in dimensions.iter().enumerate() {
        let axis_number = axis + 1;
        let dimension = i16::try_from(dimension)
            .map_err(|_| invalid(format!("axis {axis_number} is too large for NIfTI-1")))?;
        endian.write_i16(header, 42 + axis * 2, dimension);
    }
    for axis in 4..8 {
        endian.write_i16(header, 40 + axis * 2, 1);
    }
    Ok(())
}

fn write_nifti_1_srow(header: &mut [u8], rows: [[f64; 4]; 3], endian: Endian) -> Result<()> {
    for (row_index, row) in rows.iter().enumerate() {
        for (column, &value) in row.iter().enumerate() {
            endian.write_f32(
                header,
                280 + (row_index * 4 + column) * 4,
                narrow_f32(value, "srow")?,
            );
        }
    }
    Ok(())
}

fn extension_byte_count(extensions: &[NiftiExtension], cancel: &CancellationToken) -> Result<u64> {
    extensions.iter().try_fold(0_u64, |total, extension| {
        cancel.check()?;
        validate_extension(extension)?;
        total
            .checked_add((extension.payload.len() + 8) as u64)
            .ok_or(AppError::InputLimit)
    })
}

fn validate_extension(extension: &NiftiExtension) -> Result<()> {
    if extension.code < 0 {
        return Err(invalid(format!(
            "extension code {} is negative",
            extension.code
        )));
    }
    let size = extension
        .payload
        .len()
        .checked_add(8)
        .ok_or(AppError::InputLimit)?;
    if size < 16 || size % 16 != 0 || i32::try_from(size).is_err() {
        return Err(invalid(format!(
            "extension size {size} is not representable as a positive multiple of 16"
        )));
    }
    Ok(())
}

fn write_extensions(
    writer: &mut dyn Write,
    extensions: &[NiftiExtension],
    endian: Endian,
    cancel: &CancellationToken,
) -> Result<()> {
    for extension in extensions {
        validate_extension(extension)?;
        let mut header = [0_u8; 8];
        endian.write_i32(
            &mut header,
            0,
            i32::try_from(extension.payload.len() + 8)
                .map_err(|_| invalid("extension is too large"))?,
        );
        endian.write_i32(&mut header, 4, extension.code);
        write_all_checked(writer, &header, "extension header", cancel)?;
        for chunk in extension.payload.chunks(IO_CHUNK_SIZE) {
            write_all_checked(writer, chunk, "extension payload", cancel)?;
        }
    }
    Ok(())
}

fn write_zero_padding(
    writer: &mut dyn Write,
    mut bytes: u64,
    cancel: &CancellationToken,
) -> Result<()> {
    let buffer = [0_u8; IO_CHUNK_SIZE];
    while bytes > 0 {
        let amount =
            usize::try_from(bytes.min(IO_CHUNK_SIZE as u64)).map_err(|_| AppError::InputLimit)?;
        write_all_checked(
            writer,
            &buffer[..amount],
            "padding before the voxel payload",
            cancel,
        )?;
        bytes -= amount as u64;
    }
    Ok(())
}

fn write_fixed_field(
    header: &mut [u8],
    offset: usize,
    length: usize,
    value: &[u8],
    field: &str,
) -> Result<()> {
    if value.len() > length {
        return Err(invalid(format!(
            "{field} contains {} bytes but its header field allows {length}",
            value.len()
        )));
    }
    header[offset..offset + value.len()].copy_from_slice(value);
    Ok(())
}

fn narrow_i16(value: i32, field: &str) -> Result<i16> {
    i16::try_from(value).map_err(|_| invalid(format!("{field} does not fit a NIfTI-1 header")))
}

#[allow(clippy::cast_possible_truncation)]
fn narrow_f32(value: f64, field: &str) -> Result<f32> {
    let narrowed = value as f32;
    if value.is_finite() && !narrowed.is_finite() {
        Err(invalid(format!("{field} does not fit a NIfTI-1 header")))
    } else {
        Ok(narrowed)
    }
}

fn allocate_zeroed(length: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| AppError::Allocation { bytes: length })?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn read_vec_limited(
    reader: &mut dyn Read,
    consumed: &mut u64,
    length: usize,
    context: &str,
    cancel: &CancellationToken,
) -> Result<Vec<u8>> {
    cancel.check()?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| AppError::Allocation { bytes: length })?;
    let mut buffer = [0_u8; IO_CHUNK_SIZE];
    while output.len() < length {
        let amount = (length - output.len()).min(IO_CHUNK_SIZE);
        read_exact_limited(reader, consumed, &mut buffer[..amount], context, cancel)?;
        output.extend_from_slice(&buffer[..amount]);
    }
    Ok(output)
}

fn read_exact_limited(
    reader: &mut dyn Read,
    consumed: &mut u64,
    mut output: &mut [u8],
    context: &str,
    cancel: &CancellationToken,
) -> Result<()> {
    while !output.is_empty() {
        cancel.check()?;
        let available = MAX_INPUT_BYTES.saturating_sub(*consumed);
        if available == 0 {
            return Err(AppError::InputLimit);
        }
        let amount = usize::try_from(available.min(output.len() as u64))
            .map_err(|_| AppError::InputLimit)?;
        match reader.read(&mut output[..amount]) {
            Ok(0) => return Err(invalid(format!("the {context} is truncated"))),
            Ok(read) => {
                *consumed += read as u64;
                output = &mut output[read..];
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(invalid(format!("the {context} could not be read: {error}")));
            }
        }
    }
    Ok(())
}

fn drain_to_eof(
    reader: &mut dyn Read,
    consumed: &mut u64,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut buffer = [0_u8; IO_CHUNK_SIZE];
    loop {
        cancel.check()?;
        let available = MAX_INPUT_BYTES.saturating_sub(*consumed);
        let amount = usize::try_from(available.min(IO_CHUNK_SIZE as u64))
            .map_err(|_| AppError::InputLimit)?;
        let target = if amount == 0 {
            &mut buffer[..1]
        } else {
            &mut buffer[..amount]
        };
        match reader.read(target) {
            Ok(0) => return Ok(()),
            Ok(_read) if amount == 0 => return Err(AppError::InputLimit),
            Ok(read) => *consumed += read as u64,
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(invalid(format!(
                    "the input stream could not be completed: {error}"
                )));
            }
        }
    }
}

fn write_all_checked(
    writer: &mut dyn Write,
    mut bytes: &[u8],
    context: &str,
    cancel: &CancellationToken,
) -> Result<()> {
    while !bytes.is_empty() {
        cancel.check()?;
        match writer.write(bytes) {
            Ok(0) => return Err(invalid(format!("the {context} could not be written"))),
            Ok(written) => bytes = &bytes[written..],
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(invalid(format!(
                    "the {context} could not be written: {error}"
                )));
            }
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AppError {
    AppError::invalid(FORMAT, message)
}

fn unsupported(message: impl Into<String>) -> AppError {
    AppError::UnsupportedFormat(message.into())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use std::io::Cursor;

    use super::*;
    use crate::codecs::Compression;

    fn variant(version: NiftiVersion, compression: Compression) -> SourceVariant {
        SourceVariant::Nifti {
            version,
            compression,
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn fixture(
        version: NiftiVersion,
        endian: Endian,
        datatype: i16,
        bitpix: i16,
        voxel_count: usize,
        payload: &[u8],
        slope: f64,
        intercept: f64,
    ) -> Vec<u8> {
        let (header_size, prefix_size) = match version {
            NiftiVersion::One => (NIFTI_1_HEADER_SIZE, NIFTI_1_PREFIX_SIZE),
            NiftiVersion::Two => (NIFTI_2_HEADER_SIZE, NIFTI_2_PREFIX_SIZE),
        };
        let voxel_offset = prefix_size + 16;
        let mut bytes = vec![0_u8; voxel_offset + payload.len()];
        endian.write_i32(&mut bytes, 0, header_size as i32);
        match version {
            NiftiVersion::One => {
                bytes[344..348].copy_from_slice(&NIFTI_1_MAGIC);
                endian.write_i16(&mut bytes, 40, 3);
                endian.write_i16(&mut bytes, 42, voxel_count as i16);
                endian.write_i16(&mut bytes, 44, 1);
                endian.write_i16(&mut bytes, 46, 1);
                for axis in 4..8 {
                    endian.write_i16(&mut bytes, 40 + axis * 2, 1);
                }
                endian.write_i16(&mut bytes, 68, 12);
                endian.write_i16(&mut bytes, 70, datatype);
                endian.write_i16(&mut bytes, 72, bitpix);
                for (index, value) in [-1.0_f32, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]
                    .into_iter()
                    .enumerate()
                {
                    endian.write_f32(&mut bytes, 76 + index * 4, value);
                }
                endian.write_f32(&mut bytes, 108, voxel_offset as f32);
                endian.write_f32(&mut bytes, 112, slope as f32);
                endian.write_f32(&mut bytes, 116, intercept as f32);
                bytes[123] = 2;
                endian.write_f32(&mut bytes, 132, 0.25);
                endian.write_f32(&mut bytes, 136, 1.5);
                bytes[148..167].copy_from_slice(b"fixture description");
                bytes[228..239].copy_from_slice(b"fixture aux");
                endian.write_i16(&mut bytes, 252, 1);
                endian.write_i16(&mut bytes, 254, 2);
                for (index, value) in [0.1_f32, 0.2, 0.3].into_iter().enumerate() {
                    endian.write_f32(&mut bytes, 256 + index * 4, value);
                }
                for (index, value) in [4.0_f32, 5.0, 6.0].into_iter().enumerate() {
                    endian.write_f32(&mut bytes, 268 + index * 4, value);
                }
                for (index, value) in [
                    1.0_f32, 0.25, 0.0, 10.0, 0.0, 2.0, 0.5, 20.0, 0.0, 0.0, 3.0, 30.0,
                ]
                .into_iter()
                .enumerate()
                {
                    endian.write_f32(&mut bytes, 280 + index * 4, value);
                }
                bytes[328..342].copy_from_slice(b"fixture intent");
            }
            NiftiVersion::Two => {
                bytes[4..12].copy_from_slice(&NIFTI_2_MAGIC);
                endian.write_i16(&mut bytes, 12, datatype);
                endian.write_i16(&mut bytes, 14, bitpix);
                endian.write_i64(&mut bytes, 16, 3);
                endian.write_i64(&mut bytes, 24, voxel_count as i64);
                endian.write_i64(&mut bytes, 32, 1);
                endian.write_i64(&mut bytes, 40, 1);
                for axis in 4..8 {
                    endian.write_i64(&mut bytes, 16 + axis * 8, 1);
                }
                for (index, value) in [-1.0_f64, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]
                    .into_iter()
                    .enumerate()
                {
                    endian.write_f64(&mut bytes, 104 + index * 8, value);
                }
                endian.write_i64(&mut bytes, 168, voxel_offset as i64);
                endian.write_f64(&mut bytes, 176, slope);
                endian.write_f64(&mut bytes, 184, intercept);
                endian.write_f64(&mut bytes, 208, 0.25);
                endian.write_f64(&mut bytes, 216, 1.5);
                bytes[240..259].copy_from_slice(b"fixture description");
                bytes[320..331].copy_from_slice(b"fixture aux");
                endian.write_i32(&mut bytes, 344, 1);
                endian.write_i32(&mut bytes, 348, 2);
                for (index, value) in [0.1_f64, 0.2, 0.3].into_iter().enumerate() {
                    endian.write_f64(&mut bytes, 352 + index * 8, value);
                }
                for (index, value) in [4.0_f64, 5.0, 6.0].into_iter().enumerate() {
                    endian.write_f64(&mut bytes, 376 + index * 8, value);
                }
                for (index, value) in [
                    1.0_f64, 0.25, 0.0, 10.0, 0.0, 2.0, 0.5, 20.0, 0.0, 0.0, 3.0, 30.0,
                ]
                .into_iter()
                .enumerate()
                {
                    endian.write_f64(&mut bytes, 400 + index * 8, value);
                }
                endian.write_i32(&mut bytes, 500, 2);
                endian.write_i32(&mut bytes, 504, 12);
                bytes[508..522].copy_from_slice(b"fixture intent");
            }
        }
        bytes[header_size] = 1;
        endian.write_i32(&mut bytes, prefix_size, 16);
        endian.write_i32(&mut bytes, prefix_size + 4, 6);
        bytes[prefix_size + 8..prefix_size + 16].copy_from_slice(b"ABCD\0\0\0\0");
        bytes[voxel_offset..].copy_from_slice(payload);
        bytes
    }

    fn decode(bytes: &[u8], version: NiftiVersion) -> Result<DecodedVolume> {
        NIFTI_CODEC.decode(
            &mut Cursor::new(bytes),
            DecodeOptions {
                source_variant: variant(version, Compression::None),
            },
            &CancellationToken::default(),
        )
    }

    fn byte_payload_reference(
        payload: &[u8],
        signed: bool,
        slope: f64,
        intercept: f64,
    ) -> (Vec<u8>, bool) {
        let mut occupancy = Vec::with_capacity(payload.len());
        let mut normalized_non_binary = false;
        for &byte in payload {
            let raw = if signed {
                f64::from(i8::from_ne_bytes([byte]))
            } else {
                f64::from(byte)
            };
            let effective = if slope == 0.0 {
                raw
            } else {
                raw * slope + intercept
            };
            occupancy.push(u8::from(effective != 0.0));
            normalized_non_binary |= effective != 0.0 && effective != 1.0;
        }
        (occupancy, normalized_non_binary)
    }

    #[test]
    fn byte_payload_fast_path_matches_scalar_semantics() {
        for (version, endian) in [
            (NiftiVersion::One, Endian::Little),
            (NiftiVersion::Two, Endian::Big),
        ] {
            for (datatype, signed, payload) in [
                (2, false, [0, 1, 1, 0]),
                (2, false, [0, 1, 2, u8::MAX]),
                (256, true, [0, 1, 1, 0]),
                (
                    256,
                    true,
                    [0, 1, (-1_i8).cast_unsigned(), i8::MIN.cast_unsigned()],
                ),
            ] {
                for (slope, intercept) in [(1.0, 0.0), (0.0, 37.0), (0.0, f64::NAN), (1.0, 1.0)] {
                    let expected = byte_payload_reference(&payload, signed, slope, intercept);
                    let decoded = decode(
                        &fixture(
                            version,
                            endian,
                            datatype,
                            8,
                            payload.len(),
                            &payload,
                            slope,
                            intercept,
                        ),
                        version,
                    )
                    .expect("byte payload fixture should decode");
                    assert_eq!(
                        (decoded.occupancy, decoded.normalized_non_binary),
                        expected,
                        "{version:?} {endian:?} datatype={datatype} slope={slope} intercept={intercept}"
                    );
                }
            }
        }
    }

    #[test]
    fn decodes_versions_endianness_scaling_extensions_and_sform_first() {
        for (version, endian) in [
            (NiftiVersion::One, Endian::Little),
            (NiftiVersion::One, Endian::Big),
            (NiftiVersion::Two, Endian::Little),
            (NiftiVersion::Two, Endian::Big),
        ] {
            let mut payload = vec![0_u8; 8];
            for (index, value) in [0_i16, 2, 256, -1].into_iter().enumerate() {
                endian.write_i16(&mut payload, index * 2, value);
            }
            let decoded = decode(
                &fixture(version, endian, 4, 16, 4, &payload, 2.0, -4.0),
                version,
            )
            .expect("fixture should decode");
            assert_eq!(decoded.occupancy, [1, 0, 1, 1]);
            assert!(decoded.normalized_non_binary);
            assert_eq!(decoded.geometry.dimensions, [4, 1, 1]);
            assert_eq!(decoded.geometry.units, SpatialUnits::Millimeter);
            assert_eq!(
                decoded.geometry.index_to_world,
                [
                    [1.0, 0.25, 0.0, 10.0],
                    [0.0, 2.0, 0.5, 20.0],
                    [0.0, 0.0, 3.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            );
            let CodecSaveState::Nifti(metadata) = decoded.save_state else {
                panic!("expected NIfTI metadata");
            };
            assert_eq!(metadata.endian, endian);
            assert_eq!(metadata.extensions.len(), 1);
            assert_eq!(metadata.extensions[0].code, 6);
            assert_eq!(metadata.extensions[0].payload, b"ABCD\0\0\0\0");
            assert_eq!(metadata.description, b"fixture description");
            assert_eq!(metadata.auxiliary_file, b"fixture aux");
            assert_eq!(metadata.intent_name, b"fixture intent");
        }
    }

    #[test]
    fn supports_every_required_scalar_datatype() {
        for (datatype, bitpix, payload) in [
            (2, 8, vec![0, 2]),
            (256, 8, vec![0, (-2_i8).cast_unsigned()]),
            (
                4,
                16,
                two_value_payload(2, |bytes, offset| {
                    Endian::Big.write_i16(bytes, offset, -2);
                }),
            ),
            (
                512,
                16,
                two_value_payload(2, |bytes, offset| {
                    Endian::Big.write_i16(bytes, offset, 2);
                }),
            ),
            (
                8,
                32,
                two_value_payload(4, |bytes, offset| {
                    Endian::Big.write_i32(bytes, offset, -2);
                }),
            ),
            (
                768,
                32,
                two_value_payload(4, |bytes, offset| {
                    Endian::Big.write_u32(bytes, offset, 2);
                }),
            ),
            (
                1024,
                64,
                two_value_payload(8, |bytes, offset| {
                    Endian::Big.write_i64(bytes, offset, -2);
                }),
            ),
            (
                1280,
                64,
                two_value_payload(8, |bytes, offset| {
                    Endian::Big.write_u64(bytes, offset, u64::MAX);
                }),
            ),
            (
                16,
                32,
                two_value_payload(4, |bytes, offset| {
                    Endian::Big.write_f32(bytes, offset, f32::NAN);
                }),
            ),
            (
                64,
                64,
                two_value_payload(8, |bytes, offset| {
                    Endian::Big.write_f64(bytes, offset, f64::INFINITY);
                }),
            ),
        ] {
            let decoded = decode(
                &fixture(
                    NiftiVersion::Two,
                    Endian::Big,
                    datatype,
                    bitpix,
                    2,
                    &payload,
                    0.0,
                    f64::NAN,
                ),
                NiftiVersion::Two,
            )
            .expect("required scalar datatype should decode");
            assert_eq!(decoded.occupancy, [0, 1], "datatype {datatype}");
            assert!(decoded.normalized_non_binary, "datatype {datatype}");
        }
    }

    fn two_value_payload(
        scalar_bytes: usize,
        write_second: impl FnOnce(&mut [u8], usize),
    ) -> Vec<u8> {
        let mut payload = vec![0_u8; scalar_bytes * 2];
        write_second(&mut payload, scalar_bytes);
        payload
    }

    #[test]
    fn canonical_round_trip_preserves_semantic_metadata() {
        for (version, endian) in [
            (NiftiVersion::One, Endian::Big),
            (NiftiVersion::Two, Endian::Little),
        ] {
            let original = decode(
                &fixture(version, endian, 2, 8, 4, &[0, 1, 0, 1], 1.0, 0.0),
                version,
            )
            .expect("fixture should decode");
            let original_metadata = match &original.save_state {
                CodecSaveState::Nifti(metadata) => metadata.clone(),
                _ => panic!("expected NIfTI metadata"),
            };
            let replacement = [1, 0, 1, 0];
            let mut encoded = Vec::new();
            NIFTI_CODEC
                .encode(
                    VolumeView {
                        dimensions: original.geometry.dimensions,
                        occupancy: &replacement,
                        geometry: &original.geometry,
                    },
                    &original.save_state,
                    &mut encoded,
                    EncodeOptions {
                        destination_variant: variant(version, Compression::None),
                    },
                    &CancellationToken::default(),
                )
                .expect("fixture should encode");
            let reopened = decode(&encoded, version).expect("encoded fixture should reopen");
            assert_eq!(reopened.occupancy, replacement);
            assert!(!reopened.normalized_non_binary);
            assert_eq!(reopened.geometry, original.geometry);
            let CodecSaveState::Nifti(reopened_metadata) = reopened.save_state else {
                panic!("expected NIfTI metadata");
            };
            assert_eq!(reopened_metadata, original_metadata);
            let datatype_offset = if version == NiftiVersion::One { 70 } else { 12 };
            let bitpix_offset = if version == NiftiVersion::One { 72 } else { 14 };
            assert_eq!(endian.read_i16(&encoded, datatype_offset), 2);
            assert_eq!(endian.read_i16(&encoded, bitpix_offset), 8);
        }
    }

    #[test]
    #[allow(clippy::suboptimal_flops)]
    fn geometry_uses_qform_then_spacing_when_sform_is_unavailable() {
        let mut qform = fixture(NiftiVersion::One, Endian::Little, 2, 8, 1, &[1], 1.0, 0.0);
        Endian::Little.write_i16(&mut qform, 254, 0);
        let qform_decoded = decode(&qform, NiftiVersion::One).expect("qform should decode");
        let linear = &qform_decoded.geometry.index_to_world;
        let determinant = linear[0][0]
            * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
            - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
            + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0]);
        assert!(determinant < 0.0, "negative qfac must be retained");

        Endian::Little.write_i16(&mut qform, 252, 0);
        let fallback = decode(&qform, NiftiVersion::One).expect("fallback should decode");
        assert_eq!(
            fallback.geometry.index_to_world,
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
    }

    #[test]
    fn header_version_replaces_the_predecode_hint() {
        let bytes = fixture(NiftiVersion::Two, Endian::Little, 2, 8, 1, &[1], 1.0, 0.0);
        let decoded = NIFTI_CODEC
            .decode(
                &mut Cursor::new(bytes),
                DecodeOptions {
                    source_variant: variant(NiftiVersion::One, Compression::Gzip),
                },
                &CancellationToken::default(),
            )
            .expect("the header, not the path hint, identifies NIfTI-2");
        assert_eq!(
            decoded.source_variant,
            variant(NiftiVersion::Two, Compression::Gzip)
        );
    }

    #[test]
    fn accepts_defaulted_and_unaligned_single_file_offsets() {
        let mut defaulted = fixture(NiftiVersion::One, Endian::Little, 2, 8, 1, &[1], 1.0, 0.0);
        defaulted.drain(NIFTI_1_PREFIX_SIZE..NIFTI_1_PREFIX_SIZE + 16);
        defaulted[NIFTI_1_HEADER_SIZE] = 0;
        Endian::Little.write_f32(&mut defaulted, 108, 0.0);
        assert_eq!(
            decode(&defaulted, NiftiVersion::One)
                .expect("an unset single-file offset defaults to the prefix")
                .occupancy,
            [1]
        );

        let mut unaligned = defaulted;
        unaligned.insert(NIFTI_1_PREFIX_SIZE, 0x5a);
        Endian::Little.write_f32(&mut unaligned, 108, 353.0);
        assert_eq!(
            decode(&unaligned, NiftiVersion::One)
                .expect("16-byte alignment is recommended, not mandatory")
                .occupancy,
            [1]
        );
    }

    #[test]
    fn rejects_detached_multivolume_and_malformed_extensions() {
        let mut detached = fixture(NiftiVersion::One, Endian::Little, 2, 8, 1, &[1], 1.0, 0.0);
        detached[344..348].copy_from_slice(&NIFTI_1_PAIR_MAGIC);
        assert!(matches!(
            decode(&detached, NiftiVersion::One),
            Err(AppError::UnsupportedFormat(_))
        ));

        let mut multivolume = fixture(NiftiVersion::Two, Endian::Big, 2, 8, 1, &[1], 1.0, 0.0);
        Endian::Big.write_i64(&mut multivolume, 16, 4);
        Endian::Big.write_i64(&mut multivolume, 48, 2);
        assert!(matches!(
            decode(&multivolume, NiftiVersion::Two),
            Err(AppError::UnsupportedFormat(_))
        ));

        let mut hidden_multivolume =
            fixture(NiftiVersion::One, Endian::Little, 2, 8, 1, &[1], 1.0, 0.0);
        Endian::Little.write_i16(&mut hidden_multivolume, 48, 2);
        assert!(matches!(
            decode(&hidden_multivolume, NiftiVersion::One),
            Err(AppError::UnsupportedFormat(_))
        ));

        let mut bad_extension = fixture(NiftiVersion::One, Endian::Big, 2, 8, 1, &[1], 1.0, 0.0);
        Endian::Big.write_i32(&mut bad_extension, NIFTI_1_PREFIX_SIZE, 15);
        assert!(matches!(
            decode(&bad_extension, NiftiVersion::One),
            Err(AppError::InvalidFormat { .. })
        ));

        let mut missing_extension =
            fixture(NiftiVersion::Two, Endian::Little, 2, 8, 1, &[1], 1.0, 0.0);
        missing_extension[NIFTI_2_PREFIX_SIZE..NIFTI_2_PREFIX_SIZE + 8].fill(0);
        assert!(matches!(
            decode(&missing_extension, NiftiVersion::Two),
            Err(AppError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_datatypes_scaling_and_truncated_payloads() {
        for (datatype, bitpix, payload_bytes) in
            [(1, 1, 1), (32, 64, 8), (128, 24, 3), (1536, 128, 16)]
        {
            let bytes = fixture(
                NiftiVersion::One,
                Endian::Little,
                datatype,
                bitpix,
                1,
                &vec![0; payload_bytes],
                1.0,
                0.0,
            );
            assert!(matches!(
                decode(&bytes, NiftiVersion::One),
                Err(AppError::UnsupportedFormat(_))
            ));
        }

        let invalid_slope = fixture(NiftiVersion::Two, Endian::Big, 2, 8, 1, &[1], f64::NAN, 0.0);
        assert!(matches!(
            decode(&invalid_slope, NiftiVersion::Two),
            Err(AppError::InvalidFormat { .. })
        ));

        let mut truncated = fixture(
            NiftiVersion::One,
            Endian::Little,
            512,
            16,
            1,
            &[0, 1],
            1.0,
            0.0,
        );
        truncated.pop();
        assert!(matches!(
            decode(&truncated, NiftiVersion::One),
            Err(AppError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn cancellation_prevents_decode_and_encode_work() {
        let bytes = fixture(NiftiVersion::One, Endian::Little, 2, 8, 1, &[1], 1.0, 0.0);
        let decoded = decode(&bytes, NiftiVersion::One).expect("fixture should decode");
        let cancel = CancellationToken::default();
        cancel.cancel();
        let result = NIFTI_CODEC.decode(
            &mut Cursor::new(&bytes),
            DecodeOptions {
                source_variant: variant(NiftiVersion::One, Compression::None),
            },
            &cancel,
        );
        assert!(matches!(result, Err(AppError::Cancelled)));

        let mut encoded = Vec::new();
        let result = NIFTI_CODEC.encode(
            VolumeView {
                dimensions: decoded.geometry.dimensions,
                occupancy: &decoded.occupancy,
                geometry: &decoded.geometry,
            },
            &decoded.save_state,
            &mut encoded,
            EncodeOptions {
                destination_variant: variant(NiftiVersion::One, Compression::None),
            },
            &cancel,
        );
        assert!(matches!(result, Err(AppError::Cancelled)));
        assert!(encoded.is_empty());
    }

    #[test]
    fn large_nifti_1_offsets_round_up_to_a_representable_value() {
        let minimum = 268_435_472;
        let represented = representable_nifti_1_offset(minimum);
        assert!(represented >= minimum);
        assert_eq!(representable_nifti_1_offset(represented), represented);
    }
}
