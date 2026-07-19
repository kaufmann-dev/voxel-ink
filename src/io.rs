use std::{
    fs::{self, File, Metadata},
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
    time::SystemTime,
};

use flate2::{Compression as GzipLevel, read::GzDecoder, write::GzEncoder};
use sha2::{Digest, Sha256};

use crate::{
    AppError, Result,
    cancel::CancellationToken,
    codecs::{
        Compression, DecodeOptions, DecodedVolume, EncodeOptions, NiftiVersion, SourceVariant,
        codec_for_family,
    },
    error::MAX_INPUT_BYTES,
    volume::{SourceIdentity, VolumeDocument},
};

const COPY_BUFFER: usize = 1024 * 1024;

#[derive(Debug)]
pub struct LoadedVolume {
    pub decoded: DecodedVolume,
    pub identity: SourceIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveMode {
    Overwrite,
    SaveAs,
    Copy,
}

#[derive(Debug)]
pub struct SaveOutcome {
    pub identity: SourceIdentity,
    pub variant: SourceVariant,
    pub mode: SaveMode,
}

pub fn load_path(path: &Path, cancel: &CancellationToken) -> Result<LoadedVolume> {
    cancel.check()?;
    let canonical = path
        .canonicalize()
        .map_err(|source| AppError::io(path, source))?;
    let before = fs::metadata(&canonical).map_err(|source| AppError::io(&canonical, source))?;
    validate_regular_size(&canonical, &before)?;

    let hint = variant_from_path(&canonical, None)?;
    let file = File::open(&canonical).map_err(|source| AppError::io(&canonical, source))?;
    let mut source_hasher = Sha256::new();
    let hashing = HashingReader::new(file, &mut source_hasher);
    let mut buffered = BufReader::with_capacity(COPY_BUFFER, hashing);
    let prefix = buffered
        .fill_buf()
        .map_err(|source| AppError::io(&canonical, source))?;
    if prefix.len() < 2 {
        return Err(AppError::io(
            &canonical,
            io::Error::new(io::ErrorKind::UnexpectedEof, "file has no format signature"),
        ));
    }
    let gzip_magic = prefix[..2] == [0x1f, 0x8b];
    let expects_gzip = matches!(
        hint,
        SourceVariant::Nifti {
            compression: Compression::Gzip,
            ..
        }
    );
    if gzip_magic != expects_gzip {
        return Err(AppError::invalid(
            "NIfTI",
            "filename compression and stream signature do not match",
        ));
    }

    let decoded;
    if expects_gzip {
        let mut decoder = GzDecoder::new(buffered);
        decoded = decode_bounded(&mut decoder, hint, cancel)?;
        buffered = decoder.into_inner();
    } else {
        decoded = decode_bounded(&mut buffered, hint, cancel)?;
    }
    drain_reader(&mut buffered, &canonical, cancel)?;
    drop(buffered);
    let hash: [u8; 32] = source_hasher.finalize().into();

    let after = fs::metadata(&canonical).map_err(|source| AppError::io(&canonical, source))?;
    let after_hash = hash_path(&canonical, cancel)?;
    if file_fingerprint(&before) != file_fingerprint(&after) || hash != after_hash {
        return Err(AppError::SourceChanged(canonical));
    }
    Ok(LoadedVolume {
        identity: SourceIdentity {
            canonical_path: canonical,
            size: after.len(),
            modified: after.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            sha256: after_hash,
        },
        decoded,
    })
}

pub fn save_document(
    document: &VolumeDocument,
    destination: &Path,
    mode: SaveMode,
    cancel: &CancellationToken,
) -> Result<SaveOutcome> {
    cancel.check()?;
    let source_variant = document.source_variant();
    let destination_variant = variant_from_path(destination, Some(source_variant))?;
    if destination_variant.family() != source_variant.family() {
        return Err(AppError::ConversionNotAllowed(
            destination.display().to_string(),
        ));
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let canonical_parent = parent
        .canonicalize()
        .map_err(|source| AppError::io(parent, source))?;
    let file_name = destination.file_name().ok_or_else(|| {
        AppError::InvalidDimensions("save destination must have a file name".into())
    })?;
    let final_path = canonical_parent.join(file_name);
    let targets_source = final_path.canonicalize().map_or_else(
        |_| final_path == document.source_identity().canonical_path,
        |path| path == document.source_identity().canonical_path,
    );
    if mode == SaveMode::Copy && targets_source {
        return Err(AppError::InvalidSaveDestination(
            "Save a Copy must use a path other than the current source".into(),
        ));
    }
    if mode == SaveMode::Overwrite && !targets_source {
        return Err(AppError::InvalidSaveDestination(
            "Overwrite must target the current source; use Save As for another path".into(),
        ));
    }
    let mut temporary = tempfile::Builder::new()
        .prefix(".voxel-ink-save-")
        .tempfile_in(&canonical_parent)
        .map_err(|source| AppError::io(&canonical_parent, source))?;
    let mut output_hasher = Sha256::new();
    let mut output_bytes = 0_u64;
    {
        let file = temporary.as_file_mut();
        let hashing = HashingWriter::new(file, &mut output_hasher, &mut output_bytes);
        let mut buffered = BufWriter::with_capacity(COPY_BUFFER, hashing);
        if matches!(
            destination_variant,
            SourceVariant::Nifti {
                compression: Compression::Gzip,
                ..
            }
        ) {
            let mut gzip = GzEncoder::new(buffered, GzipLevel::default());
            codec_for_family(destination_variant.family()).encode(
                document.view(),
                document.save_state(),
                &mut gzip,
                EncodeOptions {
                    destination_variant,
                },
                cancel,
            )?;
            buffered = gzip
                .finish()
                .map_err(|source| AppError::io(&final_path, source))?;
        } else {
            codec_for_family(destination_variant.family()).encode(
                document.view(),
                document.save_state(),
                &mut buffered,
                EncodeOptions {
                    destination_variant,
                },
                cancel,
            )?;
        }
        buffered
            .flush()
            .map_err(|source| AppError::io(&final_path, source))?;
    }
    cancel.check()?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| AppError::io(&final_path, source))?;
    let output_hash = output_hasher.finalize().into();
    let output_metadata = temporary
        .as_file()
        .metadata()
        .map_err(|source| AppError::io(&final_path, source))?;
    if output_metadata.len() != output_bytes {
        return Err(AppError::io(
            &final_path,
            io::Error::other("saved byte count does not match the temporary file"),
        ));
    }
    if targets_source {
        verify_source_unchanged(document.source_identity(), cancel)?;
    }
    cancel.check()?;
    temporary
        .persist(&final_path)
        .map_err(|error| AppError::io(&final_path, error.error))?;
    File::open(&canonical_parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| AppError::io(&canonical_parent, source))?;
    let identity = SourceIdentity {
        canonical_path: final_path,
        size: output_metadata.len(),
        modified: output_metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        sha256: output_hash,
    };
    Ok(SaveOutcome {
        identity,
        variant: destination_variant,
        mode,
    })
}

pub fn verify_source_unchanged(
    expected: &SourceIdentity,
    cancel: &CancellationToken,
) -> Result<()> {
    let path = &expected.canonical_path;
    let current = match path.canonicalize() {
        Ok(current) if current == *path => current,
        _ => return Err(AppError::SourceChanged(path.clone())),
    };
    let metadata = fs::metadata(&current).map_err(|_| AppError::SourceChanged(path.clone()))?;
    if metadata.len() != expected.size
        || metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH) != expected.modified
    {
        return Err(AppError::SourceChanged(path.clone()));
    }
    if hash_path(&current, cancel)? != expected.sha256 {
        return Err(AppError::SourceChanged(path.clone()));
    }
    Ok(())
}

pub fn hash_path(path: &Path, cancel: &CancellationToken) -> Result<[u8; 32]> {
    let mut reader = File::open(path).map_err(|source| AppError::io(path, source))?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(COPY_BUFFER)
        .map_err(|_| AppError::Allocation { bytes: COPY_BUFFER })?;
    buffer.resize(COPY_BUFFER, 0);
    let mut hasher = Sha256::new();
    loop {
        cancel.check()?;
        let count = reader
            .read(&mut buffer)
            .map_err(|source| AppError::io(path, source))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().into())
}

fn drain_reader(reader: &mut impl Read, path: &Path, cancel: &CancellationToken) -> Result<()> {
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        cancel.check()?;
        let count = reader
            .read(&mut buffer)
            .map_err(|source| AppError::io(path, source))?;
        if count == 0 {
            return Ok(());
        }
    }
}

struct HashingReader<'a, R> {
    inner: R,
    hasher: &'a mut Sha256,
}

impl<'a, R> HashingReader<'a, R> {
    fn new(inner: R, hasher: &'a mut Sha256) -> Self {
        Self { inner, hasher }
    }
}

impl<R: Read> Read for HashingReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..read]);
        Ok(read)
    }
}

struct HashingWriter<'a, W> {
    inner: W,
    hasher: &'a mut Sha256,
    bytes_written: &'a mut u64,
}

impl<'a, W> HashingWriter<'a, W> {
    fn new(inner: W, hasher: &'a mut Sha256, bytes_written: &'a mut u64) -> Self {
        Self {
            inner,
            hasher,
            bytes_written,
        }
    }
}

impl<W: Write> Write for HashingWriter<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        *self.bytes_written = self
            .bytes_written
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::other("saved byte count overflowed u64"))?;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub fn variant_from_path(path: &Path, source: Option<SourceVariant>) -> Result<SourceVariant> {
    let lower = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| AppError::UnsupportedFormat(path.display().to_string()))?;
    if lower.ends_with(".nii.gz") {
        return Ok(SourceVariant::Nifti {
            version: nifti_version(source),
            compression: Compression::Gzip,
        });
    }
    if lower.ends_with(".nii") {
        return Ok(SourceVariant::Nifti {
            version: nifti_version(source),
            compression: Compression::None,
        });
    }
    if lower.ends_with(".binvox") {
        return Ok(SourceVariant::Binvox {
            version: match source {
                Some(SourceVariant::Binvox { version }) => version,
                _ => 1,
            },
        });
    }
    if lower.ends_with(".vox") {
        return Ok(SourceVariant::Vox);
    }
    Err(AppError::UnsupportedFormat(path.display().to_string()))
}

fn nifti_version(source: Option<SourceVariant>) -> NiftiVersion {
    match source {
        Some(SourceVariant::Nifti { version, .. }) => version,
        _ => NiftiVersion::One,
    }
}

fn validate_regular_size(path: &Path, metadata: &Metadata) -> Result<()> {
    if !metadata.is_file() {
        return Err(AppError::InvalidDimensions(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > MAX_INPUT_BYTES {
        return Err(AppError::InputLimit);
    }
    Ok(())
}

fn file_fingerprint(metadata: &Metadata) -> (u64, SystemTime) {
    (
        metadata.len(),
        metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
    )
}

fn decode_bounded<R: Read>(
    reader: R,
    hint: SourceVariant,
    cancel: &CancellationToken,
) -> Result<DecodedVolume> {
    let mut limited = LimitReader::new(reader, MAX_INPUT_BYTES);
    let result = codec_for_family(hint.family()).decode(
        &mut limited,
        DecodeOptions {
            source_variant: hint,
        },
        cancel,
    );
    if limited.exceeded {
        return Err(AppError::InputLimit);
    }
    result
}

struct LimitReader<R> {
    inner: R,
    remaining: u64,
    exceeded: bool,
}

impl<R> LimitReader<R> {
    const fn new(inner: R, limit: u64) -> Self {
        Self {
            inner,
            remaining: limit,
            exceeded: false,
        }
    }
}

impl<R: Read> Read for LimitReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut probe = [0_u8; 1];
            if self.inner.read(&mut probe)? == 0 {
                return Ok(0);
            }
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "decompressed input limit exceeded",
            ));
        }
        let allowed = usize::try_from(self.remaining)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        let count = self.inner.read(&mut buffer[..allowed])?;
        self.remaining -= count as u64;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn extensions_select_only_supported_families() {
        assert!(matches!(
            variant_from_path(Path::new("mask.NII.GZ"), None),
            Ok(SourceVariant::Nifti {
                compression: Compression::Gzip,
                ..
            })
        ));
        assert!(matches!(
            variant_from_path(Path::new("mask.binvox"), None),
            Ok(SourceVariant::Binvox { .. })
        ));
        assert!(variant_from_path(Path::new("mask.nrrd"), None).is_err());
    }

    #[test]
    fn bounded_reader_reports_an_extra_byte() {
        let mut reader = LimitReader::new(Cursor::new([1, 2, 3]), 2);
        let mut buffer = [0; 3];
        assert_eq!(reader.read(&mut buffer).unwrap(), 2);
        assert_eq!(
            reader.read(&mut buffer).unwrap_err().kind(),
            io::ErrorKind::FileTooLarge
        );
        assert!(reader.exceeded);
    }

    #[test]
    fn hashing_reader_covers_a_partially_consumed_stream_exactly_once() {
        let bytes = (0_u8..=u8::MAX).collect::<Vec<_>>();
        let expected: [u8; 32] = Sha256::digest(&bytes).into();
        let mut hasher = Sha256::new();
        {
            let hashing = HashingReader::new(Cursor::new(&bytes), &mut hasher);
            let mut buffered = BufReader::with_capacity(17, hashing);
            let mut decoded_prefix = [0_u8; 23];
            buffered.read_exact(&mut decoded_prefix).unwrap();
            assert_eq!(decoded_prefix, bytes[..23]);
            drain_reader(
                &mut buffered,
                Path::new("hashing-reader-test"),
                &CancellationToken::default(),
            )
            .unwrap();
        }
        let actual: [u8; 32] = hasher.finalize().into();
        assert_eq!(actual, expected);
    }
}
