use serde::Serialize;
use std::{fmt, io};

pub(crate) type HostResult<T> = Result<T, HostError>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HostError {
    pub(crate) code: &'static str,
    pub(crate) message: String,
}

impl HostError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self::new("invalid_request", message)
    }

    pub(crate) fn invalid_ticket() -> Self {
        Self::new(
            "invalid_ticket",
            "The file authorization ticket is invalid or has already been used.",
        )
    }

    pub(crate) fn unauthorized_source() -> Self {
        Self::new(
            "unauthorized_source",
            "The source authorization is invalid or no longer available.",
        )
    }

    pub(crate) fn source_not_ready() -> Self {
        Self::new(
            "source_not_ready",
            "The selected source has not been read successfully.",
        )
    }

    pub(crate) fn external_change() -> Self {
        Self::new(
            "external_change",
            "The destination changed outside VoxelInk. Reopen it or choose another destination.",
        )
    }

    pub(crate) fn size_limit(message: impl Into<String>) -> Self {
        Self::new("size_limit", message)
    }

    pub(crate) fn io(action: &'static str, error: &io::Error) -> Self {
        Self::new("io", format!("{action} failed ({:?}).", error.kind()))
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new("internal", message)
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for HostError {}

#[cfg(test)]
mod tests {
    use super::HostError;

    #[test]
    fn serialized_io_errors_do_not_include_paths() {
        let secret_path = "/private/example/patient.nii.gz";
        let io_error = std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("cannot open {secret_path}"),
        );
        let serialized =
            serde_json::to_string(&HostError::io("Reading the selected file", &io_error))
                .expect("serialize host error");

        assert!(serialized.contains("PermissionDenied"));
        assert!(!serialized.contains(secret_path));
    }
}
