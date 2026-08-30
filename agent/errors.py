class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""
    pass


class OneShotReturnTimeoutError(TimeoutError):
    """A one-shot run exhausted its post-tool model-return budget.

    Tool execution is deliberately outside this timeout.  The exception only
    covers the model call that follows a completed tool batch, and carries the
    durable session identifier needed to resume the run.
    """

    def __init__(self, *, session_id: str, timeout_seconds: float) -> None:
        self.session_id = str(session_id or "unknown")
        self.timeout_seconds = float(timeout_seconds)
        timeout_label = f"{self.timeout_seconds:g}"
        super().__init__(
            "final response timed out after "
            f"{timeout_label}s; task state is preserved in session "
            f"{self.session_id}. Resume with: hermes --resume {self.session_id}"
        )
