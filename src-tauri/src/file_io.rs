use crate::error::{HostError, HostResult};
use atomic_write_file::AtomicWriteFile;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, Metadata},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

pub(crate) const MAX_SOURCE_BYTES: u64 = 1_200_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FileFingerprint {
    Missing,
    File {
        size: u64,
        modified: Option<SystemTime>,
        sha256: [u8; 32],
    },
}

impl FileFingerprint {
    pub(crate) fn size(&self) -> Option<u64> {
        match self {
            Self::Missing => None,
            Self::File { size, .. } => Some(*size),
        }
    }

    pub(crate) fn same_content(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Missing, Self::Missing) => true,
            (
                Self::File {
                    size: left_size,
                    sha256: left_hash,
                    ..
                },
                Self::File {
                    size: right_size,
                    sha256: right_hash,
                    ..
                },
            ) => left_size == right_size && left_hash == right_hash,
            _ => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn hash(&self) -> Option<[u8; 32]> {
        match self {
            Self::Missing => None,
            Self::File { sha256, .. } => Some(*sha256),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ReadFile {
    pub(crate) bytes: Vec<u8>,
    pub(crate) fingerprint: FileFingerprint,
}

pub(crate) fn canonicalize_existing(path: &Path) -> HostResult<PathBuf> {
    fs::canonicalize(path).map_err(|error| HostError::io("Authorizing the selected file", &error))
}

pub(crate) fn canonicalize_destination(path: &Path) -> HostResult<PathBuf> {
    match fs::symlink_metadata(path) {
        Ok(_) => canonicalize_existing(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path.parent().ok_or_else(|| {
                HostError::invalid_request("The selected destination has no parent directory.")
            })?;
            let file_name = path.file_name().ok_or_else(|| {
                HostError::invalid_request("The selected destination has no file name.")
            })?;
            let canonical_parent = fs::canonicalize(parent)
                .map_err(|error| HostError::io("Authorizing the destination directory", &error))?;
            Ok(canonical_parent.join(file_name))
        }
        Err(error) => Err(HostError::io("Authorizing the destination", &error)),
    }
}

pub(crate) fn read_file(path: &Path) -> HostResult<ReadFile> {
    let file =
        File::open(path).map_err(|error| HostError::io("Reading the selected file", &error))?;
    let before = file
        .metadata()
        .map_err(|error| HostError::io("Inspecting the selected file", &error))?;
    require_regular_file(&before)?;
    require_source_size(before.len())?;

    let mut reader = BufReader::new(file);
    let mut bytes = Vec::with_capacity(capacity_for(before.len()));
    {
        let mut limited = reader.by_ref().take(MAX_SOURCE_BYTES + 1);
        limited
            .read_to_end(&mut bytes)
            .map_err(|error| HostError::io("Reading the selected file", &error))?;
    }
    require_source_size(bytes.len() as u64)?;

    let after = reader
        .get_ref()
        .metadata()
        .map_err(|error| HostError::io("Inspecting the selected file", &error))?;
    if metadata_changed(&before, &after) || after.len() != bytes.len() as u64 {
        return Err(HostError::external_change());
    }

    Ok(ReadFile {
        fingerprint: fingerprint_from_bytes(&after, &bytes),
        bytes,
    })
}

pub(crate) fn fingerprint(path: &Path) -> HostResult<FileFingerprint> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(FileFingerprint::Missing);
        }
        Err(error) => return Err(HostError::io("Checking the destination", &error)),
    };
    let before = file
        .metadata()
        .map_err(|error| HostError::io("Checking the destination", &error))?;
    require_regular_file(&before)?;

    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| HostError::io("Checking the destination", &error))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    let after = reader
        .get_ref()
        .metadata()
        .map_err(|error| HostError::io("Checking the destination", &error))?;
    if metadata_changed(&before, &after) {
        return Err(HostError::external_change());
    }

    Ok(FileFingerprint::File {
        size: after.len(),
        modified: after.modified().ok(),
        sha256: hasher.finalize().into(),
    })
}

pub(crate) fn atomic_replace(
    path: &Path,
    bytes: &[u8],
    expected: &FileFingerprint,
) -> HostResult<FileFingerprint> {
    let mut output = AtomicWriteFile::open(path)
        .map_err(|error| HostError::io("Preparing the destination", &error))?;
    output
        .write_all(bytes)
        .map_err(|error| HostError::io("Writing the destination", &error))?;
    output
        .sync_all()
        .map_err(|error| HostError::io("Synchronizing the destination", &error))?;

    let actual = fingerprint(path)?;
    if !expected.same_content(&actual) {
        let _ = output.discard();
        return Err(HostError::external_change());
    }

    output
        .commit()
        .map_err(|error| HostError::io("Replacing the destination", &error))?;
    let metadata =
        fs::metadata(path).map_err(|error| HostError::io("Inspecting the saved file", &error))?;
    Ok(fingerprint_from_bytes(&metadata, bytes))
}

pub(crate) fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "volume".to_owned())
}

pub(crate) fn variant_from_path(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    if name.ends_with(".nii.gz") {
        return Some("nii.gz".to_owned());
    }
    path.extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
}

pub(crate) fn path_has_variant(path: &Path, variant: &str) -> bool {
    let Some(name) = path.file_name() else {
        return false;
    };
    name.to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(&format!(".{variant}"))
}

fn fingerprint_from_bytes(metadata: &Metadata, bytes: &[u8]) -> FileFingerprint {
    FileFingerprint::File {
        size: bytes.len() as u64,
        modified: metadata.modified().ok(),
        sha256: Sha256::digest(bytes).into(),
    }
}

fn metadata_changed(before: &Metadata, after: &Metadata) -> bool {
    before.len() != after.len() || before.modified().ok() != after.modified().ok()
}

fn require_regular_file(metadata: &Metadata) -> HostResult<()> {
    if metadata.is_file() {
        Ok(())
    } else {
        Err(HostError::invalid_request(
            "The selected item is not a regular file.",
        ))
    }
}

pub(crate) fn require_source_size(size: u64) -> HostResult<()> {
    if size <= MAX_SOURCE_BYTES {
        Ok(())
    } else {
        Err(HostError::size_limit(format!(
            "The selected file exceeds the {} byte local processing limit.",
            MAX_SOURCE_BYTES
        )))
    }
}

fn capacity_for(size: u64) -> usize {
    usize::try_from(size).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn recognizes_compound_nifti_extension_first() {
        assert_eq!(
            variant_from_path(Path::new("MASK.NII.GZ")),
            Some("nii.gz".to_owned())
        );
        assert!(path_has_variant(Path::new("MASK.NII.GZ"), "nii.gz"));
        assert!(!path_has_variant(Path::new("MASK.NII.GZ"), "nii"));
    }

    #[test]
    fn hashes_a_file_without_exposing_its_contents() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("mask.nii.gz");
        fs::write(&path, b"voxel data").expect("write fixture");

        let observed = fingerprint(&path).expect("fingerprint");
        assert_eq!(observed.size(), Some(10));
        assert_eq!(observed.hash(), Some(Sha256::digest(b"voxel data").into()));
    }

    #[test]
    fn atomic_replace_commits_complete_contents() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("mask.nii.gz");
        fs::write(&path, b"old").expect("write fixture");
        let expected = fingerprint(&path).expect("fingerprint");

        let saved = atomic_replace(&path, b"complete replacement", &expected).expect("replace");

        assert_eq!(
            fs::read(&path).expect("read result"),
            b"complete replacement"
        );
        assert_eq!(saved.size(), Some(20));
    }

    #[test]
    fn external_change_aborts_without_touching_external_contents() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("mask.nii.gz");
        fs::write(&path, b"source").expect("write fixture");
        let expected = fingerprint(&path).expect("fingerprint");
        fs::write(&path, b"edited").expect("same-length external change");

        let error = atomic_replace(&path, b"editor contents", &expected).expect_err("must reject");

        assert_eq!(error.code, "external_change");
        assert_eq!(fs::read(&path).expect("read result"), b"edited");
    }

    #[test]
    fn missing_destination_is_created_atomically() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("copy.nii.gz");

        atomic_replace(&path, b"copy", &FileFingerprint::Missing).expect("create");

        assert_eq!(fs::read(path).expect("read result"), b"copy");
    }

    #[test]
    fn destination_created_after_authorization_is_not_overwritten() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("copy.nii.gz");
        fs::write(&path, b"created elsewhere").expect("external creation");

        let error = atomic_replace(&path, b"editor copy", &FileFingerprint::Missing)
            .expect_err("must reject");

        assert_eq!(error.code, "external_change");
        assert_eq!(fs::read(path).expect("read result"), b"created elsewhere");
    }

    #[test]
    fn oversized_source_is_rejected_before_allocating_its_contents() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("oversized.nii");
        File::create(&path)
            .and_then(|file| file.set_len(MAX_SOURCE_BYTES + 1))
            .expect("create sparse oversized fixture");

        let error = read_file(&path).expect_err("oversized source must fail");

        assert_eq!(error.code, "size_limit");
    }
}
