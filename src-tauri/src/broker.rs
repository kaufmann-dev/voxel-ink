use crate::{
    error::{HostError, HostResult},
    file_io::{FileFingerprint, display_name, require_source_size, variant_from_path},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf};
use uuid::Uuid;

pub(crate) const SAVE_TICKET_HEADER: &str = "x-voxelink-ticket";
pub(crate) const OPEN_EXTENSIONS: &[&str] = &["nii.gz", "nii", "nrrd", "binvox", "vox"];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum SaveMode {
    OverwriteSource,
    SaveAsAdopt,
    SaveCopy,
}

impl SaveMode {
    pub(crate) fn adopts_destination(self) -> bool {
        matches!(self, Self::OverwriteSource | Self::SaveAsAdopt)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SaveTicketRequest {
    pub(crate) mode: SaveMode,
    pub(crate) source_id: String,
    #[serde(default)]
    pub(crate) suggested_name: Option<String>,
    #[serde(default)]
    pub(crate) allowed_extensions: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OpenTicketInfo {
    pub(crate) ticket: String,
    pub(crate) source_id: String,
    pub(crate) display_name: String,
    pub(crate) source_variant: String,
    pub(crate) size: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SaveTicketInfo {
    pub(crate) ticket: String,
    pub(crate) mode: SaveMode,
    pub(crate) display_name: String,
    pub(crate) destination_variant: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WriteSaveResult {
    pub(crate) mode: SaveMode,
    pub(crate) adopted: bool,
    pub(crate) source_id: String,
    pub(crate) display_name: String,
    pub(crate) source_variant: String,
    pub(crate) destination_variant: String,
    pub(crate) size: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceRecord {
    pub(crate) path: PathBuf,
    pub(crate) display_name: String,
    pub(crate) source_variant: String,
    pub(crate) fingerprint: Option<FileFingerprint>,
}

#[derive(Clone, Debug)]
pub(crate) struct OpenTicket {
    pub(crate) source_id: Uuid,
    pub(crate) path: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct SaveTicket {
    pub(crate) source_id: Uuid,
    pub(crate) mode: SaveMode,
    pub(crate) destination: PathBuf,
    pub(crate) display_name: String,
    pub(crate) destination_variant: String,
    pub(crate) expected: FileFingerprint,
}

#[derive(Default, Debug)]
pub(crate) struct Broker {
    sources: HashMap<Uuid, SourceRecord>,
    pending_open: Option<(Uuid, OpenTicket)>,
    pending_save: Option<(Uuid, SaveTicket)>,
}

impl Broker {
    pub(crate) fn issue_open(&mut self, path: PathBuf, size: u64) -> HostResult<OpenTicketInfo> {
        require_source_size(size)?;
        let source_variant = variant_from_path(&path).ok_or_else(|| {
            HostError::invalid_request("The selected file has no supported extension.")
        })?;
        if !OPEN_EXTENSIONS.contains(&source_variant.as_str()) {
            return Err(HostError::invalid_request(
                "The selected file has no supported extension.",
            ));
        }

        if let Some((_, stale)) = self.pending_open.take() {
            self.sources.remove(&stale.source_id);
        }

        let source_id = Uuid::new_v4();
        let ticket_id = Uuid::new_v4();
        let display_name = display_name(&path);
        self.sources.insert(
            source_id,
            SourceRecord {
                path: path.clone(),
                display_name: display_name.clone(),
                source_variant: source_variant.clone(),
                fingerprint: None,
            },
        );
        self.pending_open = Some((ticket_id, OpenTicket { source_id, path }));

        Ok(OpenTicketInfo {
            ticket: ticket_id.to_string(),
            source_id: source_id.to_string(),
            display_name,
            source_variant,
            size,
        })
    }

    pub(crate) fn take_open(&mut self, ticket: &str) -> HostResult<OpenTicket> {
        let ticket_id = parse_ticket(ticket)?;
        let Some((pending_id, _)) = self.pending_open.as_ref() else {
            return Err(HostError::invalid_ticket());
        };
        if *pending_id != ticket_id {
            return Err(HostError::invalid_ticket());
        }
        Ok(self
            .pending_open
            .take()
            .expect("pending open ticket checked above")
            .1)
    }

    pub(crate) fn complete_open(
        &mut self,
        source_id: Uuid,
        fingerprint: FileFingerprint,
    ) -> HostResult<()> {
        let source = self
            .sources
            .get_mut(&source_id)
            .ok_or_else(HostError::unauthorized_source)?;
        source.fingerprint = Some(fingerprint);
        Ok(())
    }

    pub(crate) fn fail_open(&mut self, source_id: Uuid) {
        self.sources.remove(&source_id);
    }

    pub(crate) fn release_source(&mut self, source_id: &str) -> HostResult<()> {
        let source_id = parse_source(source_id)?;
        self.sources.remove(&source_id);

        if self
            .pending_open
            .as_ref()
            .is_some_and(|(_, ticket)| ticket.source_id == source_id)
        {
            self.pending_open = None;
        }
        if self
            .pending_save
            .as_ref()
            .is_some_and(|(_, ticket)| ticket.source_id == source_id)
        {
            self.pending_save = None;
        }

        Ok(())
    }

    pub(crate) fn source_for_save(
        &self,
        request: &SaveTicketRequest,
    ) -> HostResult<(Uuid, SourceRecord, Vec<String>, String)> {
        let source_id = parse_source(&request.source_id)?;
        let source = self
            .sources
            .get(&source_id)
            .cloned()
            .ok_or_else(HostError::unauthorized_source)?;
        if source.fingerprint.is_none() {
            return Err(HostError::source_not_ready());
        }

        let allowed_extensions =
            normalize_extensions(&request.allowed_extensions, &source.source_variant)?;
        let suggested_name = validate_suggested_name(
            request
                .suggested_name
                .as_deref()
                .unwrap_or(&source.display_name),
        )?;

        Ok((source_id, source, allowed_extensions, suggested_name))
    }

    pub(crate) fn issue_save(
        &mut self,
        source_id: Uuid,
        mode: SaveMode,
        destination: PathBuf,
        destination_variant: String,
        expected: FileFingerprint,
    ) -> SaveTicketInfo {
        let ticket_id = Uuid::new_v4();
        let display_name = display_name(&destination);
        self.pending_save = Some((
            ticket_id,
            SaveTicket {
                source_id,
                mode,
                destination,
                display_name: display_name.clone(),
                destination_variant: destination_variant.clone(),
                expected,
            },
        ));
        SaveTicketInfo {
            ticket: ticket_id.to_string(),
            mode,
            display_name,
            destination_variant,
        }
    }

    pub(crate) fn take_save(&mut self, ticket: &str) -> HostResult<SaveTicket> {
        let ticket_id = parse_ticket(ticket)?;
        let Some((pending_id, _)) = self.pending_save.as_ref() else {
            return Err(HostError::invalid_ticket());
        };
        if *pending_id != ticket_id {
            return Err(HostError::invalid_ticket());
        }
        Ok(self
            .pending_save
            .take()
            .expect("pending save ticket checked above")
            .1)
    }

    pub(crate) fn complete_save(
        &mut self,
        ticket: &SaveTicket,
        fingerprint: FileFingerprint,
    ) -> HostResult<WriteSaveResult> {
        let source_variant = if ticket.mode.adopts_destination() {
            ticket.destination_variant.clone()
        } else {
            self.sources
                .get(&ticket.source_id)
                .ok_or_else(HostError::unauthorized_source)?
                .source_variant
                .clone()
        };

        if ticket.mode.adopts_destination() {
            let source = self
                .sources
                .get_mut(&ticket.source_id)
                .ok_or_else(HostError::unauthorized_source)?;
            source.path = ticket.destination.clone();
            source.display_name = ticket.display_name.clone();
            source.source_variant = ticket.destination_variant.clone();
            source.fingerprint = Some(fingerprint.clone());
        }

        Ok(WriteSaveResult {
            mode: ticket.mode,
            adopted: ticket.mode.adopts_destination(),
            source_id: ticket.source_id.to_string(),
            display_name: ticket.display_name.clone(),
            source_variant,
            destination_variant: ticket.destination_variant.clone(),
            size: fingerprint.size().unwrap_or(0),
        })
    }

    #[cfg(test)]
    pub(crate) fn source(&self, source_id: &str) -> Option<&SourceRecord> {
        Uuid::parse_str(source_id)
            .ok()
            .and_then(|source_id| self.sources.get(&source_id))
    }
}

pub(crate) fn same_format_family(left: &str, right: &str) -> bool {
    matches!(
        (left, right),
        ("nii", "nii") | ("nii", "nii.gz") | ("nii.gz", "nii") | ("nii.gz", "nii.gz")
    ) || left == right
}

fn normalize_extensions(requested: &[String], source_variant: &str) -> HostResult<Vec<String>> {
    if requested.len() > OPEN_EXTENSIONS.len() {
        return Err(HostError::invalid_request(
            "Too many destination extensions were requested.",
        ));
    }

    let candidates: Vec<&str> = if requested.is_empty() {
        vec![source_variant]
    } else {
        requested.iter().map(String::as_str).collect()
    };
    let mut normalized = Vec::with_capacity(candidates.len());
    for extension in candidates {
        let extension = extension
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase();
        if !OPEN_EXTENSIONS.contains(&extension.as_str())
            || !same_format_family(source_variant, &extension)
        {
            return Err(HostError::invalid_request(
                "Save As must stay within the source format family.",
            ));
        }
        if !normalized.contains(&extension) {
            normalized.push(extension);
        }
    }
    Ok(normalized)
}

fn validate_suggested_name(name: &str) -> HostResult<String> {
    let name = name.trim();
    if name.is_empty()
        || name.len() > 255
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
    {
        return Err(HostError::invalid_request(
            "The suggested destination name is invalid.",
        ));
    }
    Ok(name.to_owned())
}

fn parse_ticket(ticket: &str) -> HostResult<Uuid> {
    Uuid::parse_str(ticket).map_err(|_| HostError::invalid_ticket())
}

fn parse_source(source_id: &str) -> HostResult<Uuid> {
    Uuid::parse_str(source_id).map_err(|_| HostError::unauthorized_source())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn ready_fingerprint(contents: &[u8]) -> FileFingerprint {
        FileFingerprint::File {
            size: contents.len() as u64,
            modified: None,
            sha256: Sha256::digest(contents).into(),
        }
    }

    fn issue_ready_source(broker: &mut Broker) -> OpenTicketInfo {
        let info = broker
            .issue_open(PathBuf::from("/authorized/patient.nii.gz"), 7)
            .expect("issue open");
        let ticket = broker.take_open(&info.ticket).expect("consume open");
        broker
            .complete_open(ticket.source_id, ready_fingerprint(b"fixture"))
            .expect("complete open");
        info
    }

    #[test]
    fn open_ticket_is_one_use() {
        let mut broker = Broker::default();
        let info = broker
            .issue_open(PathBuf::from("/authorized/patient.nii.gz"), 12)
            .expect("issue open");

        broker.take_open(&info.ticket).expect("first use");
        let error = broker
            .take_open(&info.ticket)
            .expect_err("second use must fail");

        assert_eq!(error.code, "invalid_ticket");
    }

    #[test]
    fn oversized_source_does_not_receive_authority() {
        let mut broker = Broker::default();
        let error = broker
            .issue_open(
                PathBuf::from("/authorized/oversized.nii"),
                crate::file_io::MAX_SOURCE_BYTES + 1,
            )
            .expect_err("oversized source must fail");

        assert_eq!(error.code, "size_limit");
    }

    #[test]
    fn newer_open_ticket_invalidates_the_previous_ticket() {
        let mut broker = Broker::default();
        let old = broker
            .issue_open(PathBuf::from("/authorized/old.nii.gz"), 1)
            .expect("old ticket");
        let new = broker
            .issue_open(PathBuf::from("/authorized/new.nii.gz"), 2)
            .expect("new ticket");

        assert_eq!(
            broker
                .take_open(&old.ticket)
                .expect_err("stale ticket")
                .code,
            "invalid_ticket"
        );
        assert_eq!(
            broker.take_open(&new.ticket).expect("current ticket").path,
            PathBuf::from("/authorized/new.nii.gz")
        );
    }

    #[test]
    fn source_cannot_be_saved_until_read_completes() {
        let mut broker = Broker::default();
        let info = broker
            .issue_open(PathBuf::from("/authorized/patient.nii.gz"), 12)
            .expect("issue open");
        let request = SaveTicketRequest {
            mode: SaveMode::OverwriteSource,
            source_id: info.source_id,
            suggested_name: None,
            allowed_extensions: Vec::new(),
        };

        assert_eq!(
            broker
                .source_for_save(&request)
                .expect_err("source is not ready")
                .code,
            "source_not_ready"
        );
    }

    #[test]
    fn releasing_a_source_is_idempotent_and_revokes_its_pending_tickets() {
        let mut broker = Broker::default();
        let source = issue_ready_source(&mut broker);
        let source_id = Uuid::parse_str(&source.source_id).expect("source id");
        let save = broker.issue_save(
            source_id,
            SaveMode::OverwriteSource,
            PathBuf::from("/authorized/patient.nii.gz"),
            "nii.gz".to_owned(),
            ready_fingerprint(b"fixture"),
        );

        broker
            .release_source(&source.source_id)
            .expect("first release");
        broker
            .release_source(&source.source_id)
            .expect("idempotent release");

        assert!(broker.source(&source.source_id).is_none());
        assert_eq!(
            broker
                .take_save(&save.ticket)
                .expect_err("revoked ticket")
                .code,
            "invalid_ticket"
        );
    }

    #[test]
    fn releasing_a_source_revokes_an_unread_open_ticket() {
        let mut broker = Broker::default();
        let source = broker
            .issue_open(PathBuf::from("/authorized/patient.nii.gz"), 7)
            .expect("issue open");

        broker
            .release_source(&source.source_id)
            .expect("release source");

        assert_eq!(
            broker
                .take_open(&source.ticket)
                .expect_err("revoked ticket")
                .code,
            "invalid_ticket"
        );
    }

    #[test]
    fn save_ticket_is_one_use_and_serialization_never_contains_paths() {
        let mut broker = Broker::default();
        let source = issue_ready_source(&mut broker);
        let source_id = Uuid::parse_str(&source.source_id).expect("source id");
        let info = broker.issue_save(
            source_id,
            SaveMode::SaveCopy,
            PathBuf::from("/very/private/copy.nii.gz"),
            "nii.gz".to_owned(),
            FileFingerprint::Missing,
        );

        let serialized = serde_json::to_string(&info).expect("serialize ticket info");
        assert!(!serialized.contains("/very/private"));
        assert!(serialized.contains("\"destinationVariant\":\"nii.gz\""));
        broker.take_save(&info.ticket).expect("first use");
        assert_eq!(
            broker.take_save(&info.ticket).expect_err("second use").code,
            "invalid_ticket"
        );
    }

    #[test]
    fn save_as_accepts_only_the_nifti_compression_family() {
        let mut broker = Broker::default();
        let source = issue_ready_source(&mut broker);
        let valid = SaveTicketRequest {
            mode: SaveMode::SaveAsAdopt,
            source_id: source.source_id.clone(),
            suggested_name: None,
            allowed_extensions: vec!["nii".to_owned(), "nii.gz".to_owned()],
        };
        let invalid = SaveTicketRequest {
            allowed_extensions: vec!["nrrd".to_owned()],
            ..valid.clone()
        };

        assert_eq!(
            broker.source_for_save(&valid).expect("valid family").2,
            vec!["nii".to_owned(), "nii.gz".to_owned()]
        );
        assert_eq!(
            broker
                .source_for_save(&invalid)
                .expect_err("conversion must fail")
                .code,
            "invalid_request"
        );
    }

    #[test]
    fn save_copy_does_not_adopt_the_destination() {
        let mut broker = Broker::default();
        let source_info = issue_ready_source(&mut broker);
        let source_id = Uuid::parse_str(&source_info.source_id).expect("source id");
        let original = broker
            .source(&source_info.source_id)
            .expect("source")
            .path
            .clone();
        let save_info = broker.issue_save(
            source_id,
            SaveMode::SaveCopy,
            PathBuf::from("/authorized/copy.nii.gz"),
            "nii.gz".to_owned(),
            FileFingerprint::Missing,
        );
        let ticket = broker.take_save(&save_info.ticket).expect("save ticket");

        let result = broker
            .complete_save(&ticket, ready_fingerprint(b"copy"))
            .expect("complete copy");

        assert!(!result.adopted);
        assert_eq!(result.source_variant, "nii.gz");
        assert_eq!(result.destination_variant, "nii.gz");
        assert_eq!(
            broker.source(&source_info.source_id).expect("source").path,
            original
        );
    }

    #[test]
    fn save_as_adopts_destination_and_variant() {
        let mut broker = Broker::default();
        let source_info = issue_ready_source(&mut broker);
        let source_id = Uuid::parse_str(&source_info.source_id).expect("source id");
        let save_info = broker.issue_save(
            source_id,
            SaveMode::SaveAsAdopt,
            PathBuf::from("/authorized/adopted.nii"),
            "nii".to_owned(),
            FileFingerprint::Missing,
        );
        let ticket = broker.take_save(&save_info.ticket).expect("save ticket");

        let result = broker
            .complete_save(&ticket, ready_fingerprint(b"saved"))
            .expect("complete save as");
        let source = broker.source(&source_info.source_id).expect("source");

        assert!(result.adopted);
        assert_eq!(result.source_variant, "nii");
        assert_eq!(result.destination_variant, "nii");
        assert_eq!(source.path, PathBuf::from("/authorized/adopted.nii"));
        assert_eq!(source.source_variant, "nii");
    }

    #[test]
    fn serialized_save_modes_are_explicit() {
        assert_eq!(
            serde_json::to_string(&SaveMode::OverwriteSource).expect("serialize"),
            "\"overwriteSource\""
        );
        assert_eq!(
            serde_json::to_string(&SaveMode::SaveAsAdopt).expect("serialize"),
            "\"saveAsAdopt\""
        );
        assert_eq!(
            serde_json::to_string(&SaveMode::SaveCopy).expect("serialize"),
            "\"saveCopy\""
        );
    }
}
