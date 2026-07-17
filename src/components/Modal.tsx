import { useEffect, useId, useRef, type ReactNode } from "react";

export interface ModalAction {
  readonly label: string;
  readonly onSelect: () => void;
  readonly tone?: "default" | "primary" | "danger";
  readonly autoFocus?: boolean;
}

interface ModalProps {
  readonly title: string;
  readonly children: ReactNode;
  readonly actions: readonly ModalAction[];
  readonly onCancel: () => void;
}

export function Modal({ title, children, actions, onCancel }: ModalProps) {
  const titleId = useId();
  const descriptionId = useId();
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const panel = panelRef.current;
    const focusable = panel?.querySelectorAll<HTMLElement>("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])");
    const requested = panel?.querySelector<HTMLElement>("[data-autofocus='true']");
    (requested ?? focusable?.[0])?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCancel();
        return;
      }
      if (event.key !== "Tab" || !focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [onCancel]);

  return (
    <div className="modal-backdrop" role="presentation">
      <div
        ref={panelRef}
        className="modal-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <h2 id={titleId}>{title}</h2>
        <div id={descriptionId} className="modal-description">{children}</div>
        <div className="modal-actions">
          {actions.map((action) => (
            <button
              key={action.label}
              type="button"
              className={`button ${action.tone === "primary" ? "button-primary" : action.tone === "danger" ? "button-danger" : ""}`}
              data-autofocus={action.autoFocus ? "true" : undefined}
              onClick={action.onSelect}
            >
              {action.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
