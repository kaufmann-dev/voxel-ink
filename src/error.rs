use std::path::PathBuf;

pub const MAX_VOXELS: u64 = 2_147_483_647;
pub const MAX_INPUT_BYTES: u64 = 1_200_000_000;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("operation cancelled")]
    Cancelled,
    #[error("unsupported file format: {0}")]
    UnsupportedFormat(String),
    #[error("invalid {format} file: {message}")]
    InvalidFormat {
        format: &'static str,
        message: String,
    },
    #[error("volume dimensions are invalid: {0}")]
    InvalidDimensions(String),
    #[error("volume contains {voxels} voxels; the limit is {MAX_VOXELS}")]
    VoxelLimit { voxels: u64 },
    #[error("input exceeds the {MAX_INPUT_BYTES}-byte safety limit")]
    InputLimit,
    #[error("not enough memory to allocate {bytes} bytes for the volume")]
    Allocation { bytes: usize },
    #[error("edit needs {bytes} bytes of undo history; the limit is {limit} bytes")]
    UndoHistoryLimit { bytes: usize, limit: usize },
    #[error("bridge footprint examination exceeds the {limit}-cell safety limit")]
    BridgeWorkLimit { limit: usize },
    #[error("bridge would change more than {limit} voxels")]
    BridgeVoxelLimit { limit: usize },
    #[error("the affine transform is singular or non-finite")]
    InvalidAffine,
    #[error("the source file changed outside VoxelInk: {0}")]
    SourceChanged(PathBuf),
    #[error("no volume is open")]
    NoDocument,
    #[error("this format cannot be converted to {0}")]
    ConversionNotAllowed(String),
    #[error("invalid save destination: {0}")]
    InvalidSaveDestination(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("graphics adapter unavailable: {0}")]
    Adapter(String),
    #[error("internal channel disconnected")]
    ChannelDisconnected,
    #[error("the engine command queue is busy; try again after the current operation finishes")]
    EngineBusy,
}

impl AppError {
    #[must_use]
    pub fn invalid(format: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidFormat {
            format,
            message: message.into(),
        }
    }

    #[must_use]
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, AppError>;

pub fn checked_voxel_count(dimensions: [u32; 3]) -> Result<usize> {
    if dimensions.contains(&0) {
        return Err(AppError::InvalidDimensions(
            "all three axes must be non-zero".into(),
        ));
    }
    let voxels = dimensions
        .iter()
        .try_fold(1_u64, |total, &axis| total.checked_mul(u64::from(axis)));
    let Some(voxels) = voxels else {
        return Err(AppError::VoxelLimit { voxels: u64::MAX });
    };
    if voxels > MAX_VOXELS {
        return Err(AppError::VoxelLimit { voxels });
    }
    usize::try_from(voxels).map_err(|_| AppError::VoxelLimit { voxels })
}
