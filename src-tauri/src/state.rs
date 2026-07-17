use crate::broker::Broker;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExitDecision {
    Allow,
    Prevent,
    PreventAndNotify,
}

#[derive(Debug, Default)]
pub(crate) struct HostState {
    pub(crate) broker: Arc<Mutex<Broker>>,
    exit_prompt_pending: AtomicBool,
    allow_exit_once: AtomicBool,
}

impl HostState {
    pub(crate) fn intercept_close(&self) -> ExitDecision {
        if self.exit_prompt_pending.swap(true, Ordering::AcqRel) {
            ExitDecision::Prevent
        } else {
            ExitDecision::PreventAndNotify
        }
    }

    pub(crate) fn intercept_exit(&self) -> ExitDecision {
        if self.allow_exit_once.swap(false, Ordering::AcqRel) {
            self.exit_prompt_pending.store(false, Ordering::Release);
            ExitDecision::Allow
        } else {
            self.intercept_close()
        }
    }

    pub(crate) fn confirm_exit(&self) {
        self.exit_prompt_pending.store(false, Ordering::Release);
        self.allow_exit_once.store(true, Ordering::Release);
    }

    pub(crate) fn cancel_exit(&self) {
        self.allow_exit_once.store(false, Ordering::Release);
        self.exit_prompt_pending.store(false, Ordering::Release);
    }

    pub(crate) fn event_delivery_failed(&self) {
        self.exit_prompt_pending.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_native_requests_create_only_one_prompt() {
        let state = HostState::default();

        assert_eq!(state.intercept_close(), ExitDecision::PreventAndNotify);
        assert_eq!(state.intercept_close(), ExitDecision::Prevent);
        assert_eq!(state.intercept_exit(), ExitDecision::Prevent);
    }

    #[test]
    fn cancel_allows_a_later_request_to_prompt_again() {
        let state = HostState::default();
        assert_eq!(state.intercept_close(), ExitDecision::PreventAndNotify);

        state.cancel_exit();

        assert_eq!(state.intercept_close(), ExitDecision::PreventAndNotify);
    }

    #[test]
    fn confirmation_allows_exactly_one_exit_event() {
        let state = HostState::default();
        assert_eq!(state.intercept_close(), ExitDecision::PreventAndNotify);
        state.confirm_exit();

        assert_eq!(state.intercept_exit(), ExitDecision::Allow);
        assert_eq!(state.intercept_exit(), ExitDecision::PreventAndNotify);
    }
}
