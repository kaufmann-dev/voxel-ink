use std::io::{BufReader, ErrorKind, Read, Write};

use crate::{
    AppError, Result,
    cancel::CancellationToken,
    codecs::{
        BinvoxMetadata, CodecSaveState, DecodeOptions, DecodedVolume, EncodeOptions, FormatFamily,
        SourceVariant, VolumeCodec,
    },
    error::{MAX_INPUT_BYTES, checked_voxel_count},
    volume::{SpatialUnits, VolumeGeometry, VolumeView},
};

const FORMAT: &str = "BINVOX";
const MAX_HEADER_BYTES: usize = 4096;
const MAX_HEADER_LINE_BYTES: usize = 1024;
const IO_BUFFER_BYTES: usize = 64 * 1024;

pub static BINVOX_CODEC: BinvoxCodec = BinvoxCodec;

#[derive(Debug)]
pub struct BinvoxCodec;

impl VolumeCodec for BinvoxCodec {
    fn family(&self) -> FormatFamily {
        FormatFamily::Binvox
    }

    fn decode(
        &self,
        reader: &mut dyn Read,
        options: DecodeOptions,
        cancel: &CancellationToken,
    ) -> Result<DecodedVolume> {
        cancel.check()?;
        // Codec-local buffering keeps tiny header and RLE reads away from the
        // dynamic reader supplied by the generic codec interface.
        let mut buffered = BufReader::with_capacity(IO_BUFFER_BYTES, reader);
        let (metadata, header_bytes) = read_header(&mut buffered)?;
        match options.source_variant {
            SourceVariant::Binvox { .. } => {}
            _ => {
                return Err(AppError::UnsupportedFormat(
                    "the source variant is not BINVOX".into(),
                ));
            }
        }

        let voxel_count = checked_voxel_count(metadata.dimensions)?;
        let mut occupancy = Vec::new();
        occupancy
            .try_reserve_exact(voxel_count)
            .map_err(|_| AppError::Allocation { bytes: voxel_count })?;
        occupancy.resize(voxel_count, 0);

        let [nx, ny, nz] = metadata.dimensions.map(u64::from);
        let yz_plane = ny
            .checked_mul(nz)
            .ok_or(AppError::VoxelLimit { voxels: u64::MAX })?;
        let mut file_index = 0_u64;
        let mut payload_bytes = 0_u64;
        let total_voxels =
            u64::try_from(voxel_count).map_err(|_| AppError::VoxelLimit { voxels: u64::MAX })?;
        let mut normalized_non_binary = false;

        while file_index < total_voxels {
            cancel.check()?;
            let [value, count] = read_payload_pair(&mut buffered)?;
            payload_bytes = payload_bytes.checked_add(2).ok_or(AppError::InputLimit)?;
            if u64::try_from(header_bytes).unwrap_or(u64::MAX) + payload_bytes > MAX_INPUT_BYTES {
                return Err(AppError::InputLimit);
            }
            if count == 0 {
                return Err(AppError::invalid(FORMAT, "a run has a zero length"));
            }
            if metadata.version == 1 && value > 1 {
                return Err(AppError::invalid(
                    FORMAT,
                    format!("version 1 run has invalid voxel value {value}"),
                ));
            }

            let end = file_index
                .checked_add(u64::from(count))
                .ok_or_else(|| AppError::invalid(FORMAT, "run length overflow"))?;
            if end > total_voxels {
                return Err(AppError::invalid(
                    FORMAT,
                    "RLE data expands beyond the declared dimensions",
                ));
            }
            if value > 1 {
                normalized_non_binary = true;
            }
            if value != 0 {
                fill_occupied_run(&mut occupancy, [nx, ny, nz], yz_plane, file_index, end)?;
            }
            file_index = end;
        }

        let mut trailing = [0_u8; 1];
        match buffered.read(&mut trailing) {
            Ok(0) => {}
            Ok(_) => {
                return Err(AppError::invalid(
                    FORMAT,
                    "trailing bytes follow the complete RLE stream",
                ));
            }
            Err(error) => return Err(stream_error("read trailing data", error)),
        }

        let geometry = geometry_from_metadata(&metadata)?;
        let version = metadata.version;
        Ok(DecodedVolume {
            occupancy,
            geometry,
            save_state: CodecSaveState::Binvox(metadata),
            source_variant: SourceVariant::Binvox { version },
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
        let CodecSaveState::Binvox(metadata) = state else {
            return Err(AppError::ConversionNotAllowed("BINVOX".into()));
        };
        let destination_version = match options.destination_variant {
            SourceVariant::Binvox { version } => version,
            _ => return Err(AppError::ConversionNotAllowed("non-BINVOX format".into())),
        };
        validate_metadata(metadata)?;
        if destination_version != metadata.version {
            return Err(AppError::ConversionNotAllowed(format!(
                "BINVOX version {destination_version}"
            )));
        }
        validate_volume(volume, metadata.dimensions)?;

        let header = format!(
            "#binvox {}\ndim {} {} {}\ntranslate {} {} {}\nscale {}\ndata\n",
            metadata.version,
            metadata.dimensions[0],
            metadata.dimensions[1],
            metadata.dimensions[2],
            metadata.translation[0],
            metadata.translation[1],
            metadata.translation[2],
            metadata.scale,
        );
        if header.len() > MAX_HEADER_BYTES {
            return Err(AppError::invalid(FORMAT, "generated header is too large"));
        }

        write_bytes(writer, header.as_bytes())?;

        let [nx, ny, nz] = metadata.dimensions.map(|extent| extent as usize);
        let mut current_value = None;
        let mut current_count = 0_u8;
        let mut run_bytes = Vec::new();
        run_bytes
            .try_reserve_exact(IO_BUFFER_BYTES)
            .map_err(|_| AppError::Allocation {
                bytes: IO_BUFFER_BYTES,
            })?;
        let mut processed = 0_usize;

        // BINVOX stores y fastest, followed by z, then x. Nested cursors avoid
        // division and modulo for every voxel while preserving that order.
        for x in 0..nx {
            for z in 0..nz {
                let mut canonical_index = x + nx * ny * z;
                for _y in 0..ny {
                    if processed.is_multiple_of(4096) {
                        cancel.check()?;
                    }
                    let value = volume.occupancy[canonical_index];
                    if current_value == Some(value) && current_count < u8::MAX {
                        current_count += 1;
                    } else {
                        if let Some(previous_value) = current_value {
                            push_encoded_run(
                                writer,
                                &mut run_bytes,
                                previous_value,
                                current_count,
                            )?;
                        }
                        current_value = Some(value);
                        current_count = 1;
                    }
                    processed += 1;
                    canonical_index += nx;
                }
            }
        }
        if let Some(value) = current_value {
            push_encoded_run(writer, &mut run_bytes, value, current_count)?;
        }
        if !run_bytes.is_empty() {
            write_bytes(writer, &run_bytes)?;
        }
        cancel.check()?;
        writer
            .flush()
            .map_err(|error| stream_error("flush encoded data", error))
    }
}

fn read_header(reader: &mut dyn Read) -> Result<(BinvoxMetadata, usize)> {
    let mut bytes_read = 0;
    let version_line = read_ascii_line(reader, &mut bytes_read)?;
    let dimensions_line = read_ascii_line(reader, &mut bytes_read)?;

    let version_fields = fields(&version_line, 2, "version")?;
    if version_fields[0] != "#binvox" {
        return Err(AppError::invalid(FORMAT, "missing #binvox signature"));
    }
    let version = parse_u32(version_fields[1], "version")?;
    if !matches!(version, 1 | 2) {
        return Err(AppError::invalid(
            FORMAT,
            format!("unsupported version {version}"),
        ));
    }

    let dimension_fields = fields(&dimensions_line, 4, "dimensions")?;
    if dimension_fields[0] != "dim" {
        return Err(AppError::invalid(FORMAT, "expected a dim header line"));
    }
    let dimensions = [
        parse_u32(dimension_fields[1], "x dimension")?,
        parse_u32(dimension_fields[2], "y dimension")?,
        parse_u32(dimension_fields[3], "z dimension")?,
    ];

    let next_line = read_ascii_line(reader, &mut bytes_read)?;
    let next_fields: Vec<_> = next_line.split_ascii_whitespace().collect();
    let (translation, scale) = match next_fields.first().copied() {
        Some("data") => {
            if next_fields.len() != 1 {
                return Err(AppError::invalid(
                    FORMAT,
                    "data marker has unexpected fields",
                ));
            }
            ([0.0; 3], 1.0)
        }
        Some("translate") => {
            if next_fields.len() != 4 {
                return Err(AppError::invalid(
                    FORMAT,
                    "translation line must contain three coordinates",
                ));
            }
            let translation = [
                parse_f64(next_fields[1], "x translation")?,
                parse_f64(next_fields[2], "y translation")?,
                parse_f64(next_fields[3], "z translation")?,
            ];
            let scale_line = read_ascii_line(reader, &mut bytes_read)?;
            let scale_fields = fields(&scale_line, 2, "scale")?;
            if scale_fields[0] != "scale" {
                return Err(AppError::invalid(
                    FORMAT,
                    "translate must be followed by scale",
                ));
            }
            let scale = parse_f64(scale_fields[1], "scale")?;
            if scale <= 0.0 {
                return Err(AppError::invalid(FORMAT, "scale must be positive"));
            }
            let data_line = read_ascii_line(reader, &mut bytes_read)?;
            let data_fields = fields(&data_line, 1, "data marker")?;
            if data_fields[0] != "data" {
                return Err(AppError::invalid(FORMAT, "expected the data marker"));
            }
            (translation, scale)
        }
        _ => {
            return Err(AppError::invalid(
                FORMAT,
                "expected translate or data after dimensions",
            ));
        }
    };

    let metadata = BinvoxMetadata {
        version,
        dimensions,
        translation,
        scale,
    };
    validate_metadata(&metadata)?;
    Ok((metadata, bytes_read))
}

fn read_ascii_line(reader: &mut dyn Read, total_bytes: &mut usize) -> Result<String> {
    let mut line = Vec::new();
    loop {
        if *total_bytes >= MAX_HEADER_BYTES {
            return Err(AppError::invalid(
                FORMAT,
                format!("header exceeds {MAX_HEADER_BYTES} bytes"),
            ));
        }
        let mut byte = [0_u8; 1];
        match reader.read(&mut byte) {
            Ok(0) => return Err(AppError::invalid(FORMAT, "header is truncated")),
            Ok(_) => {}
            Err(error) => return Err(stream_error("read header", error)),
        }
        *total_bytes += 1;
        if byte[0] == b'\n' {
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return String::from_utf8(line)
                .map_err(|_| AppError::invalid(FORMAT, "header is not ASCII"));
        }
        if !matches!(byte[0], b'\t' | b'\r' | b' '..=b'~') {
            return Err(AppError::invalid(FORMAT, "header is not printable ASCII"));
        }
        if line.len() >= MAX_HEADER_LINE_BYTES {
            return Err(AppError::invalid(
                FORMAT,
                format!("header line exceeds {MAX_HEADER_LINE_BYTES} bytes"),
            ));
        }
        line.push(byte[0]);
    }
}

fn fields<'a>(line: &'a str, expected: usize, name: &str) -> Result<Vec<&'a str>> {
    let fields: Vec<_> = line.split_ascii_whitespace().collect();
    if fields.len() != expected {
        return Err(AppError::invalid(
            FORMAT,
            format!(
                "{name} line has {} fields; expected {expected}",
                fields.len()
            ),
        ));
    }
    Ok(fields)
}

fn parse_u32(value: &str, field: &str) -> Result<u32> {
    value
        .parse()
        .map_err(|_| AppError::invalid(FORMAT, format!("invalid {field}")))
}

fn parse_f64(value: &str, field: &str) -> Result<f64> {
    let parsed: f64 = value
        .parse()
        .map_err(|_| AppError::invalid(FORMAT, format!("invalid {field}")))?;
    if !parsed.is_finite() {
        return Err(AppError::invalid(FORMAT, format!("{field} is not finite")));
    }
    Ok(parsed)
}

fn read_payload_pair(reader: &mut impl Read) -> Result<[u8; 2]> {
    let mut pair = [0_u8; 2];
    match reader.read_exact(&mut pair) {
        Ok(()) => Ok(pair),
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => Err(AppError::invalid(
            FORMAT,
            "RLE stream is truncated while reading a value/count pair",
        )),
        Err(error) => Err(stream_error("read RLE data", error)),
    }
}

fn fill_occupied_run(
    occupancy: &mut [u8],
    dimensions: [u64; 3],
    yz_plane: u64,
    start: u64,
    end: u64,
) -> Result<()> {
    let [nx, ny, nz] = dimensions;
    let mut x = start / yz_plane;
    let within_x = start % yz_plane;
    let mut z = within_x / ny;
    let mut y = within_x % ny;
    let mut canonical = x + nx * (y + ny * z);
    for _ in start..end {
        let index = usize::try_from(canonical)
            .map_err(|_| AppError::invalid(FORMAT, "voxel index cannot be represented"))?;
        occupancy[index] = 1;
        y += 1;
        if y == ny {
            y = 0;
            z += 1;
            if z == nz {
                z = 0;
                x += 1;
            }
            canonical = x + nx * ny * z;
        } else {
            canonical += nx;
        }
    }
    Ok(())
}

fn validate_metadata(metadata: &BinvoxMetadata) -> Result<()> {
    if !matches!(metadata.version, 1 | 2) {
        return Err(AppError::invalid(
            FORMAT,
            format!("unsupported version {}", metadata.version),
        ));
    }
    checked_voxel_count(metadata.dimensions)?;
    if !metadata.translation.iter().all(|value| value.is_finite()) {
        return Err(AppError::invalid(FORMAT, "translation is not finite"));
    }
    if !metadata.scale.is_finite() || metadata.scale <= 0.0 {
        return Err(AppError::invalid(
            FORMAT,
            "scale must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_volume(volume: VolumeView<'_>, expected_dimensions: [u32; 3]) -> Result<()> {
    if volume.dimensions != expected_dimensions || volume.geometry.dimensions != expected_dimensions
    {
        return Err(AppError::InvalidDimensions(
            "BINVOX save metadata and volume dimensions differ".into(),
        ));
    }
    let expected_len = checked_voxel_count(volume.dimensions)?;
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

fn geometry_from_metadata(metadata: &BinvoxMetadata) -> Result<VolumeGeometry> {
    let spacing = [
        metadata.scale / f64::from(metadata.dimensions[0]),
        metadata.scale / f64::from(metadata.dimensions[1]),
        metadata.scale / f64::from(metadata.dimensions[2]),
    ];
    VolumeGeometry::new(
        metadata.dimensions,
        [
            [
                spacing[0],
                0.0,
                0.0,
                metadata.translation[0] + spacing[0] * 0.5,
            ],
            [
                0.0,
                spacing[1],
                0.0,
                metadata.translation[1] + spacing[1] * 0.5,
            ],
            [
                0.0,
                0.0,
                spacing[2],
                metadata.translation[2] + spacing[2] * 0.5,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        SpatialUnits::Unknown,
    )
}

fn push_encoded_run(
    writer: &mut dyn Write,
    buffer: &mut Vec<u8>,
    value: u8,
    count: u8,
) -> Result<()> {
    if buffer.len() + 2 > IO_BUFFER_BYTES {
        write_bytes(writer, buffer)?;
        buffer.clear();
    }
    buffer.extend_from_slice(&[value, count]);
    Ok(())
}

fn write_bytes(writer: &mut dyn Write, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|error| stream_error("write encoded data", error))
}

fn stream_error(action: &str, error: std::io::Error) -> AppError {
    AppError::invalid(FORMAT, format!("could not {action}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn options(version: u32) -> DecodeOptions {
        DecodeOptions {
            source_variant: SourceVariant::Binvox { version },
        }
    }

    fn file(version: u32, dimensions: [u32; 3], payload: &[u8]) -> Vec<u8> {
        let mut bytes = format!(
            "#binvox {version}\ndim {} {} {}\ntranslate 1 2 3\nscale 6\ndata\n",
            dimensions[0], dimensions[1], dimensions[2]
        )
        .into_bytes();
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn decodes_y_then_z_then_x_order_into_canonical_layout() {
        let bytes = file(1, [2, 3, 2], &[0, 1, 1, 1, 0, 4, 1, 1, 0, 5]);
        let decoded = BINVOX_CODEC
            .decode(
                &mut Cursor::new(bytes),
                options(1),
                &CancellationToken::default(),
            )
            .unwrap();

        let occupied: Vec<_> = decoded
            .occupancy
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value == 1).then_some(index))
            .collect();
        assert_eq!(occupied, [1, 2]);
        assert_eq!(decoded.geometry.dimensions, [2, 3, 2]);
        assert_eq!(decoded.geometry.spacing, [3.0, 2.0, 3.0]);
        assert_eq!(
            decoded.geometry.index_point_to_world([0.0, 0.0, 0.0]),
            [2.5, 3.0, 4.5]
        );
        assert!(!decoded.normalized_non_binary);
    }

    #[test]
    fn version_two_normalizes_non_binary_values() {
        let bytes = file(2, [2, 1, 1], &[7, 1, 0, 1]);
        let decoded = BINVOX_CODEC
            .decode(
                &mut Cursor::new(bytes),
                // Path detection supplies a family hint; the header is authoritative.
                options(1),
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(decoded.occupancy, [1, 0]);
        assert!(decoded.normalized_non_binary);
        assert_eq!(
            decoded.save_state,
            CodecSaveState::Binvox(BinvoxMetadata {
                version: 2,
                dimensions: [2, 1, 1],
                translation: [1.0, 2.0, 3.0],
                scale: 6.0,
            })
        );
    }

    #[test]
    fn accepts_the_specification_optional_transform_lines() {
        let bytes = b"#binvox 1\ndim 2 1 1\ndata\n\x01\x01\x00\x01";
        let decoded = BINVOX_CODEC
            .decode(
                &mut Cursor::new(bytes),
                options(1),
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(decoded.occupancy, [1, 0]);
        assert_eq!(decoded.geometry.spacing, [0.5, 1.0, 1.0]);
        assert_eq!(
            decoded.save_state,
            CodecSaveState::Binvox(BinvoxMetadata {
                version: 1,
                dimensions: [2, 1, 1],
                translation: [0.0; 3],
                scale: 1.0,
            })
        );
    }

    #[test]
    fn rejects_invalid_and_truncated_runs() {
        for payload in [&[0, 0][..], &[0, 3][..], &[1][..]] {
            let bytes = file(1, [2, 1, 1], payload);
            assert!(
                BINVOX_CODEC
                    .decode(
                        &mut Cursor::new(bytes),
                        options(1),
                        &CancellationToken::default()
                    )
                    .is_err()
            );
        }
        let non_binary = file(1, [1, 1, 1], &[2, 1]);
        assert!(
            BINVOX_CODEC
                .decode(
                    &mut Cursor::new(non_binary),
                    options(1),
                    &CancellationToken::default()
                )
                .is_err()
        );
        let trailing = file(1, [1, 1, 1], &[1, 1, 0]);
        assert!(
            BINVOX_CODEC
                .decode(
                    &mut Cursor::new(trailing),
                    options(1),
                    &CancellationToken::default()
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_oversized_ascii_header_line() {
        let bytes = format!("#binvox 1{}\n", " ".repeat(MAX_HEADER_LINE_BYTES)).into_bytes();
        assert!(
            BINVOX_CODEC
                .decode(
                    &mut Cursor::new(bytes),
                    options(1),
                    &CancellationToken::default()
                )
                .is_err()
        );
    }

    #[test]
    fn encode_round_trip_preserves_metadata_and_occupancy() {
        let metadata = BinvoxMetadata {
            version: 2,
            dimensions: [3, 2, 2],
            translation: [-1.25, 2.5, 7.0],
            scale: 3.5,
        };
        let geometry = geometry_from_metadata(&metadata).unwrap();
        let occupancy = [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1];
        let mut encoded = Vec::new();
        BINVOX_CODEC
            .encode(
                VolumeView {
                    dimensions: metadata.dimensions,
                    occupancy: &occupancy,
                    geometry: &geometry,
                },
                &CodecSaveState::Binvox(metadata.clone()),
                &mut encoded,
                EncodeOptions {
                    destination_variant: SourceVariant::Binvox { version: 2 },
                },
                &CancellationToken::default(),
            )
            .unwrap();
        let decoded = BINVOX_CODEC
            .decode(
                &mut Cursor::new(encoded),
                options(2),
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(decoded.occupancy, occupancy);
        assert_eq!(decoded.save_state, CodecSaveState::Binvox(metadata));
    }

    #[test]
    fn cancellation_stops_decode_and_encode() {
        let cancel = CancellationToken::default();
        cancel.cancel();
        let bytes = file(1, [1, 1, 1], &[1, 1]);
        assert!(matches!(
            BINVOX_CODEC.decode(&mut Cursor::new(bytes), options(1), &cancel),
            Err(AppError::Cancelled)
        ));

        let metadata = BinvoxMetadata {
            version: 1,
            dimensions: [1, 1, 1],
            translation: [0.0; 3],
            scale: 1.0,
        };
        let geometry = geometry_from_metadata(&metadata).unwrap();
        assert!(matches!(
            BINVOX_CODEC.encode(
                VolumeView {
                    dimensions: [1, 1, 1],
                    occupancy: &[1],
                    geometry: &geometry,
                },
                &CodecSaveState::Binvox(metadata),
                &mut Vec::new(),
                EncodeOptions {
                    destination_variant: SourceVariant::Binvox { version: 1 },
                },
                &cancel,
            ),
            Err(AppError::Cancelled)
        ));
    }
}
