use std::io::{BufWriter, ErrorKind, Read, Write};

use crate::{
    AppError, Result,
    cancel::CancellationToken,
    codecs::{
        CodecSaveState, DecodeOptions, DecodedVolume, EncodeOptions, FormatFamily, SourceVariant,
        VolumeCodec, VoxMetadata,
    },
    error::{MAX_INPUT_BYTES, checked_voxel_count},
    volume::{SpatialUnits, VolumeGeometry, VolumeView},
};

const FORMAT: &str = "VOX";
const VOX_VERSION: u32 = 150;
const CHUNK_HEADER_BYTES: u64 = 12;

pub static VOX_CODEC: VoxCodec = VoxCodec;

#[derive(Debug)]
pub struct VoxCodec;

impl VolumeCodec for VoxCodec {
    fn family(&self) -> FormatFamily {
        FormatFamily::Vox
    }

    fn decode(
        &self,
        reader: &mut dyn Read,
        options: DecodeOptions,
        cancel: &CancellationToken,
    ) -> Result<DecodedVolume> {
        cancel.check()?;
        if options.source_variant != SourceVariant::Vox {
            return Err(AppError::UnsupportedFormat(
                "the source variant is not MagicaVoxel VOX".into(),
            ));
        }

        let magic = read_array::<4>(reader, "file signature")?;
        if magic != *b"VOX " {
            return Err(AppError::invalid(FORMAT, "missing VOX signature"));
        }
        let version = read_u32(reader, "file version")?;
        if version != VOX_VERSION {
            return Err(AppError::invalid(
                FORMAT,
                format!("unsupported file version {version}"),
            ));
        }

        let main = read_chunk_header(reader, "MAIN chunk")?;
        if main.id != *b"MAIN" {
            return Err(AppError::invalid(FORMAT, "the root chunk is not MAIN"));
        }
        if main.content_bytes != 0 {
            return Err(AppError::invalid(
                FORMAT,
                "MAIN chunk content must be empty",
            ));
        }
        let file_bytes = 20_u64
            .checked_add(u64::from(main.children_bytes))
            .ok_or(AppError::InputLimit)?;
        if file_bytes > MAX_INPUT_BYTES {
            return Err(AppError::InputLimit);
        }

        let mut remaining = u64::from(main.children_bytes);
        let mut dimensions = None;
        let mut occupancy = None;
        let mut xyzi_seen = false;
        let mut normalized_non_binary = false;

        while remaining != 0 {
            cancel.check()?;
            if remaining < CHUNK_HEADER_BYTES {
                return Err(AppError::invalid(
                    FORMAT,
                    "MAIN children end in a partial chunk header",
                ));
            }
            let chunk = read_chunk_header(reader, "child chunk")?;
            remaining -= CHUNK_HEADER_BYTES;
            let payload_bytes = u64::from(chunk.content_bytes)
                .checked_add(u64::from(chunk.children_bytes))
                .ok_or_else(|| AppError::invalid(FORMAT, "chunk size overflow"))?;
            if payload_bytes > remaining {
                return Err(AppError::invalid(
                    FORMAT,
                    format!(
                        "{} chunk extends beyond its MAIN parent",
                        chunk_name(chunk.id)
                    ),
                ));
            }

            match &chunk.id {
                b"PACK" => {
                    return Err(AppError::invalid(
                        FORMAT,
                        "multi-model PACK files are not supported",
                    ));
                }
                b"SIZE" => {
                    if dimensions.is_some() {
                        return Err(AppError::invalid(
                            FORMAT,
                            "multiple SIZE model chunks are not supported",
                        ));
                    }
                    if chunk.content_bytes != 12 || chunk.children_bytes != 0 {
                        return Err(AppError::invalid(
                            FORMAT,
                            "SIZE chunk must contain exactly 12 bytes and no children",
                        ));
                    }
                    let parsed_dimensions = read_dimensions(reader)?;
                    let voxel_count = checked_voxel_count(parsed_dimensions)?;
                    let mut parsed_occupancy = Vec::new();
                    parsed_occupancy
                        .try_reserve_exact(voxel_count)
                        .map_err(|_| AppError::Allocation { bytes: voxel_count })?;
                    parsed_occupancy.resize(voxel_count, 0);
                    dimensions = Some(parsed_dimensions);
                    occupancy = Some(parsed_occupancy);
                }
                b"XYZI" => {
                    if xyzi_seen {
                        return Err(AppError::invalid(
                            FORMAT,
                            "multiple XYZI model chunks are not supported",
                        ));
                    }
                    let parsed_dimensions = dimensions.ok_or_else(|| {
                        AppError::invalid(FORMAT, "XYZI chunk appears before SIZE")
                    })?;
                    let parsed_occupancy = occupancy.as_mut().ok_or_else(|| {
                        AppError::invalid(FORMAT, "internal model allocation is missing")
                    })?;
                    if chunk.children_bytes != 0 || chunk.content_bytes < 4 {
                        return Err(AppError::invalid(
                            FORMAT,
                            "XYZI chunk has an invalid size or child list",
                        ));
                    }
                    normalized_non_binary |= read_xyzi(
                        reader,
                        chunk.content_bytes,
                        parsed_dimensions,
                        parsed_occupancy,
                        cancel,
                    )?;
                    xyzi_seen = true;
                }
                _ => skip_exact(reader, payload_bytes, cancel)?,
            }
            remaining -= payload_bytes;
        }

        let dimensions = dimensions
            .ok_or_else(|| AppError::invalid(FORMAT, "file does not contain a SIZE chunk"))?;
        let occupancy = occupancy
            .ok_or_else(|| AppError::invalid(FORMAT, "file does not contain model storage"))?;
        if !xyzi_seen {
            return Err(AppError::invalid(
                FORMAT,
                "file does not contain an XYZI chunk",
            ));
        }
        let mut trailing = [0_u8; 1];
        match reader.read(&mut trailing) {
            Ok(0) => {}
            Ok(_) => {
                return Err(AppError::invalid(
                    FORMAT,
                    "trailing bytes follow the MAIN chunk",
                ));
            }
            Err(error) => return Err(stream_error("read trailing data", error)),
        }

        let geometry = identity_geometry(dimensions)?;
        Ok(DecodedVolume {
            occupancy,
            geometry,
            save_state: CodecSaveState::Vox(VoxMetadata { dimensions }),
            source_variant: SourceVariant::Vox,
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
        cancel.check()?;
        if options.destination_variant != SourceVariant::Vox {
            return Err(AppError::ConversionNotAllowed("non-VOX format".into()));
        }
        let CodecSaveState::Vox(metadata) = state else {
            return Err(AppError::ConversionNotAllowed("VOX".into()));
        };
        validate_volume(volume, metadata.dimensions)?;

        let occupied_count = volume
            .occupancy
            .iter()
            .try_fold(0_u32, |count, &value| count.checked_add(u32::from(value)))
            .ok_or_else(|| AppError::invalid(FORMAT, "occupied voxel count overflow"))?;
        let xyzi_content_bytes = occupied_count
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or_else(|| AppError::invalid(FORMAT, "XYZI chunk is too large"))?;
        let main_children_bytes = 24_u32
            .checked_add(12)
            .and_then(|bytes| bytes.checked_add(xyzi_content_bytes))
            .ok_or_else(|| AppError::invalid(FORMAT, "MAIN chunk is too large"))?;

        let mut buffered = BufWriter::with_capacity(64 * 1024, writer);
        write_bytes(&mut buffered, b"VOX ")?;
        write_u32(&mut buffered, VOX_VERSION)?;
        write_chunk_header(&mut buffered, *b"MAIN", 0, main_children_bytes)?;
        write_chunk_header(&mut buffered, *b"SIZE", 12, 0)?;
        for dimension in metadata.dimensions {
            let signed = i32::try_from(dimension)
                .map_err(|_| AppError::invalid(FORMAT, "dimension does not fit i32"))?;
            write_bytes(&mut buffered, &signed.to_le_bytes())?;
        }
        write_chunk_header(&mut buffered, *b"XYZI", xyzi_content_bytes, 0)?;
        write_u32(&mut buffered, occupied_count)?;

        let nx = usize::try_from(metadata.dimensions[0])
            .map_err(|_| AppError::invalid(FORMAT, "x dimension cannot be represented"))?;
        let ny = usize::try_from(metadata.dimensions[1])
            .map_err(|_| AppError::invalid(FORMAT, "y dimension cannot be represented"))?;
        let slice = nx
            .checked_mul(ny)
            .ok_or_else(|| AppError::invalid(FORMAT, "slice size overflow"))?;
        for (index, &value) in volume.occupancy.iter().enumerate() {
            if index % 4096 == 0 {
                cancel.check()?;
            }
            if value == 0 {
                continue;
            }
            let z = index / slice;
            let within_slice = index % slice;
            let y = within_slice / nx;
            let x = within_slice % nx;
            write_bytes(
                &mut buffered,
                &[
                    u8::try_from(x)
                        .map_err(|_| AppError::invalid(FORMAT, "x coordinate exceeds 255"))?,
                    u8::try_from(y)
                        .map_err(|_| AppError::invalid(FORMAT, "y coordinate exceeds 255"))?,
                    u8::try_from(z)
                        .map_err(|_| AppError::invalid(FORMAT, "z coordinate exceeds 255"))?,
                    1,
                ],
            )?;
        }
        cancel.check()?;
        buffered
            .flush()
            .map_err(|error| stream_error("flush encoded data", error))
    }
}

#[derive(Clone, Copy)]
struct ChunkHeader {
    id: [u8; 4],
    content_bytes: u32,
    children_bytes: u32,
}

fn read_chunk_header(reader: &mut dyn Read, field: &str) -> Result<ChunkHeader> {
    Ok(ChunkHeader {
        id: read_array(reader, field)?,
        content_bytes: read_u32(reader, field)?,
        children_bytes: read_u32(reader, field)?,
    })
}

fn read_dimensions(reader: &mut dyn Read) -> Result<[u32; 3]> {
    let mut dimensions = [0_u32; 3];
    for (axis, dimension) in dimensions.iter_mut().enumerate() {
        let raw = i32::from_le_bytes(read_array(reader, "SIZE dimensions")?);
        if !(1..=256).contains(&raw) {
            return Err(AppError::invalid(
                FORMAT,
                format!("SIZE axis {axis} must be between 1 and 256"),
            ));
        }
        *dimension =
            u32::try_from(raw).map_err(|_| AppError::invalid(FORMAT, "negative SIZE dimension"))?;
    }
    Ok(dimensions)
}

fn read_xyzi(
    reader: &mut dyn Read,
    content_bytes: u32,
    dimensions: [u32; 3],
    occupancy: &mut [u8],
    cancel: &CancellationToken,
) -> Result<bool> {
    let voxel_records = read_u32(reader, "XYZI voxel count")?;
    let expected_content = voxel_records
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(|| AppError::invalid(FORMAT, "XYZI voxel count overflow"))?;
    if content_bytes != expected_content {
        return Err(AppError::invalid(
            FORMAT,
            format!(
                "XYZI declares {content_bytes} content bytes but its count requires {expected_content}"
            ),
        ));
    }
    if usize::try_from(voxel_records).unwrap_or(usize::MAX) > occupancy.len() {
        return Err(AppError::invalid(
            FORMAT,
            "XYZI contains more records than cells in the model",
        ));
    }

    let [nx, ny, nz] = dimensions.map(usize::try_from).map(|value| {
        value.map_err(|_| AppError::invalid(FORMAT, "dimension cannot be represented"))
    });
    let nx = nx?;
    let ny = ny?;
    let nz = nz?;
    let mut normalized_non_binary = false;

    for record in 0..voxel_records {
        if record % 1024 == 0 {
            cancel.check()?;
        }
        let [x, y, z, palette_index] = read_array(reader, "XYZI voxel record")?;
        if palette_index == 0 {
            return Err(AppError::invalid(
                FORMAT,
                "XYZI palette index zero is reserved for empty voxels",
            ));
        }
        let x = usize::from(x);
        let y = usize::from(y);
        let z = usize::from(z);
        if x >= nx || y >= ny || z >= nz {
            return Err(AppError::invalid(
                FORMAT,
                format!("XYZI voxel ({x}, {y}, {z}) lies outside the SIZE bounds"),
            ));
        }
        let index = x + nx * (y + ny * z);
        if occupancy[index] != 0 {
            return Err(AppError::invalid(
                FORMAT,
                format!("XYZI contains duplicate voxel ({x}, {y}, {z})"),
            ));
        }
        occupancy[index] = 1;
        normalized_non_binary |= palette_index != 1;
    }
    Ok(normalized_non_binary)
}

fn validate_volume(volume: VolumeView<'_>, expected_dimensions: [u32; 3]) -> Result<()> {
    if volume.dimensions != expected_dimensions || volume.geometry.dimensions != expected_dimensions
    {
        return Err(AppError::InvalidDimensions(
            "VOX save metadata and volume dimensions differ".into(),
        ));
    }
    if expected_dimensions
        .iter()
        .any(|dimension| !(1..=256).contains(dimension))
    {
        return Err(AppError::InvalidDimensions(
            "VOX dimensions must be between 1 and 256".into(),
        ));
    }
    let expected_len = checked_voxel_count(expected_dimensions)?;
    if volume.occupancy.len() != expected_len {
        return Err(AppError::InvalidDimensions(format!(
            "expected {expected_len} occupancy bytes, got {}",
            volume.occupancy.len()
        )));
    }
    if volume.occupancy.iter().any(|&value| value > 1) {
        return Err(AppError::InvalidDimensions(
            "occupancy must contain only zero or one".into(),
        ));
    }
    Ok(())
}

fn identity_geometry(dimensions: [u32; 3]) -> Result<VolumeGeometry> {
    VolumeGeometry::new(
        dimensions,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        SpatialUnits::Unknown,
    )
}

fn read_array<const N: usize>(reader: &mut dyn Read, field: &str) -> Result<[u8; N]> {
    let mut bytes = [0_u8; N];
    match reader.read_exact(&mut bytes) {
        Ok(()) => Ok(bytes),
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => Err(AppError::invalid(
            FORMAT,
            format!("file is truncated while reading {field}"),
        )),
        Err(error) => Err(stream_error("read input", error)),
    }
}

fn read_u32(reader: &mut dyn Read, field: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array(reader, field)?))
}

fn skip_exact(reader: &mut dyn Read, mut bytes: u64, cancel: &CancellationToken) -> Result<()> {
    let mut scratch = [0_u8; 8192];
    while bytes != 0 {
        cancel.check()?;
        let amount = usize::try_from(bytes.min(scratch.len() as u64))
            .map_err(|_| AppError::invalid(FORMAT, "chunk size cannot be represented"))?;
        match reader.read_exact(&mut scratch[..amount]) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                return Err(AppError::invalid(
                    FORMAT,
                    "file is truncated inside an unknown chunk",
                ));
            }
            Err(error) => return Err(stream_error("skip unknown chunk", error)),
        }
        bytes -= u64::try_from(amount)
            .map_err(|_| AppError::invalid(FORMAT, "skip amount cannot be represented"))?;
    }
    Ok(())
}

fn write_chunk_header(
    writer: &mut dyn Write,
    id: [u8; 4],
    content_bytes: u32,
    children_bytes: u32,
) -> Result<()> {
    write_bytes(writer, &id)?;
    write_u32(writer, content_bytes)?;
    write_u32(writer, children_bytes)
}

fn write_u32(writer: &mut dyn Write, value: u32) -> Result<()> {
    write_bytes(writer, &value.to_le_bytes())
}

fn write_bytes(writer: &mut dyn Write, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|error| stream_error("write encoded data", error))
}

fn chunk_name(id: [u8; 4]) -> String {
    id.into_iter()
        .map(|byte| {
            if byte.is_ascii_graphic() {
                char::from(byte)
            } else {
                '?'
            }
        })
        .collect()
}

fn stream_error(action: &str, error: std::io::Error) -> AppError {
    AppError::invalid(FORMAT, format!("could not {action}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn chunk(id: [u8; 4], content: &[u8], children: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&id);
        bytes.extend_from_slice(&u32::try_from(content.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(children.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(content);
        bytes.extend_from_slice(children);
        bytes
    }

    fn size(dimensions: [i32; 3]) -> Vec<u8> {
        let mut content = Vec::new();
        for dimension in dimensions {
            content.extend_from_slice(&dimension.to_le_bytes());
        }
        chunk(*b"SIZE", &content, &[])
    }

    fn xyzi(voxels: &[[u8; 4]]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend_from_slice(&u32::try_from(voxels.len()).unwrap().to_le_bytes());
        for voxel in voxels {
            content.extend_from_slice(voxel);
        }
        chunk(*b"XYZI", &content, &[])
    }

    fn file(children: &[u8]) -> Vec<u8> {
        let mut bytes = b"VOX ".to_vec();
        bytes.extend_from_slice(&VOX_VERSION.to_le_bytes());
        bytes.extend_from_slice(&chunk(*b"MAIN", &[], children));
        bytes
    }

    fn decode(bytes: Vec<u8>) -> Result<DecodedVolume> {
        VOX_CODEC.decode(
            &mut Cursor::new(bytes),
            DecodeOptions {
                source_variant: SourceVariant::Vox,
            },
            &CancellationToken::default(),
        )
    }

    #[test]
    fn decodes_single_model_and_ignores_bounded_unknown_chunks() {
        let nested = chunk(*b"NOTE", b"nested", &[]);
        let unknown = chunk(*b"JUNK", b"ignored", &nested);
        let mut children = unknown;
        children.extend_from_slice(&size([2, 3, 4]));
        children.extend_from_slice(&xyzi(&[[1, 2, 3, 7], [0, 0, 0, 1]]));

        let decoded = decode(file(&children)).unwrap();
        assert_eq!(decoded.geometry.dimensions, [2, 3, 4]);
        assert_eq!(
            decoded
                .occupancy
                .iter()
                .filter(|&&value| value == 1)
                .count(),
            2
        );
        assert_eq!(decoded.occupancy[0], 1);
        assert_eq!(decoded.occupancy[23], 1);
        assert!(decoded.normalized_non_binary);
    }

    #[test]
    fn rejects_duplicate_and_out_of_bounds_voxels() {
        for voxels in [
            &[[0, 0, 0, 1], [0, 0, 0, 2]][..],
            &[[2, 0, 0, 1]][..],
            &[[0, 0, 0, 0]][..],
        ] {
            let mut children = size([2, 2, 2]);
            children.extend_from_slice(&xyzi(voxels));
            assert!(decode(file(&children)).is_err());
        }
    }

    #[test]
    fn rejects_pack_and_multiple_model_chunks() {
        let pack = chunk(*b"PACK", &1_u32.to_le_bytes(), &[]);
        assert!(decode(file(&pack)).is_err());

        let mut multiple = size([1, 1, 1]);
        multiple.extend_from_slice(&xyzi(&[[0, 0, 0, 1]]));
        multiple.extend_from_slice(&size([1, 1, 1]));
        multiple.extend_from_slice(&xyzi(&[]));
        assert!(decode(file(&multiple)).is_err());
    }

    #[test]
    fn rejects_invalid_chunk_bounds_and_truncation() {
        let mut children = size([2, 2, 2]);
        children.extend_from_slice(&xyzi(&[[1, 1, 1, 1]]));
        let mut truncated = file(&children);
        truncated.pop();
        assert!(decode(truncated).is_err());

        let mut invalid_size = Vec::new();
        invalid_size.extend_from_slice(b"SIZE");
        invalid_size.extend_from_slice(&100_u32.to_le_bytes());
        invalid_size.extend_from_slice(&0_u32.to_le_bytes());
        invalid_size.extend_from_slice(&[0; 12]);
        assert!(decode(file(&invalid_size)).is_err());
    }

    #[test]
    fn rejects_dimensions_outside_magica_voxel_range() {
        for dimensions in [[0, 1, 1], [257, 1, 1], [-1, 1, 1]] {
            let mut children = size(dimensions);
            children.extend_from_slice(&xyzi(&[]));
            assert!(decode(file(&children)).is_err());
        }
    }

    #[test]
    fn encode_round_trip_writes_palette_index_one() {
        let dimensions = [3, 2, 2];
        let geometry = identity_geometry(dimensions).unwrap();
        let occupancy = [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1];
        let mut encoded = Vec::new();
        VOX_CODEC
            .encode(
                VolumeView {
                    dimensions,
                    occupancy: &occupancy,
                    geometry: &geometry,
                },
                &CodecSaveState::Vox(VoxMetadata { dimensions }),
                &mut encoded,
                EncodeOptions {
                    destination_variant: SourceVariant::Vox,
                },
                &CancellationToken::default(),
            )
            .unwrap();

        let decoded = decode(encoded).unwrap();
        assert_eq!(decoded.occupancy, occupancy);
        assert!(!decoded.normalized_non_binary);
    }

    #[test]
    fn cancellation_stops_decode_and_encode() {
        let cancel = CancellationToken::default();
        cancel.cancel();
        let mut children = size([1, 1, 1]);
        children.extend_from_slice(&xyzi(&[[0, 0, 0, 1]]));
        assert!(matches!(
            VOX_CODEC.decode(
                &mut Cursor::new(file(&children)),
                DecodeOptions {
                    source_variant: SourceVariant::Vox,
                },
                &cancel
            ),
            Err(AppError::Cancelled)
        ));

        let geometry = identity_geometry([1, 1, 1]).unwrap();
        assert!(matches!(
            VOX_CODEC.encode(
                VolumeView {
                    dimensions: [1, 1, 1],
                    occupancy: &[1],
                    geometry: &geometry,
                },
                &CodecSaveState::Vox(VoxMetadata {
                    dimensions: [1, 1, 1],
                }),
                &mut Vec::new(),
                EncodeOptions {
                    destination_variant: SourceVariant::Vox,
                },
                &cancel,
            ),
            Err(AppError::Cancelled)
        ));
    }
}
