"""Thread-safe state management for document translation."""

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AdvancedTranslationState:
    """Enhanced translation state management with comprehensive processing."""

    def __init__(self):
        self.current_file: Optional[str] = None
        self.current_content: Optional[Dict[str, Any]] = None
        self.source_language: Optional[str] = None
        self.target_language: Optional[str] = None
        self.translation_progress: int = 0
        self.translation_status: str = "idle"
        self.error_message: str = ""
        self.job_id: Optional[str] = None
        self.output_file: Optional[str] = None
        self.processing_info: Dict[str, Any] = {}
        self.backup_path: Optional[str] = None
        self.max_pages: int = 0  # 0 means translate all pages
        self.session_id: Optional[str] = None
        self.neologism_analysis: Optional[Dict[str, Any]] = None
        self.user_choices: List[Dict[str, Any]] = []
        self.philosophy_mode: bool = False
        self.backup_path = None
        self.max_pages: int = 0  # 0 means translate all pages
        self.session_id: Optional[str] = None
        self.neologism_analysis: Optional[Dict[str, Any]] = None
        self.user_choices: List[Dict[str, Any]] = []
        self.philosophy_mode: bool = False


class ThreadSafeTranslationJobs:
    """Thread-safe translation job management with automatic cleanup."""

    def __init__(self, retention_hours: int = 24):
        """Initialize thread-safe job manager.

        Args:
            retention_hours: Hours to retain completed jobs before cleanup
        """
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._retention_hours = retention_hours
        self._cleanup_interval = 3600  # Run cleanup every hour
        self._last_cleanup = time.time()

    def add_job(self, job_id: str, job_data: Dict[str, Any]) -> None:
        """Add a new job with timestamp."""
        with self._lock:
            job_data["timestamp"] = datetime.now()
            self._jobs[job_id] = job_data
            self._maybe_cleanup()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job data. Returns True if job exists."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)
                return True
            return False

    def remove_job(self, job_id: str) -> bool:
        """Remove a job. Returns True if job existed."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get a copy of all jobs."""
        with self._lock:
            return dict(self._jobs)

    def __contains__(self, job_id: str) -> bool:
        """Check if job exists."""
        with self._lock:
            return job_id in self._jobs

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        """Get job data using subscript notation."""
        with self._lock:
            return self._jobs[job_id]

    def __setitem__(self, job_id: str, job_data: Dict[str, Any]) -> None:
        """Set job data using subscript notation."""
        self.add_job(job_id, job_data)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed since last cleanup."""
        current_time = time.time()
        if current_time - self._last_cleanup >= self._cleanup_interval:
            self._cleanup_old_jobs()
            self._last_cleanup = current_time

    def _cleanup_old_jobs(self) -> None:
        """Remove jobs older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self._retention_hours)
        jobs_to_remove = []

        for job_id, job_data in self._jobs.items():
            # Only cleanup completed or failed jobs
            if job_data.get("status") in ["completed", "failed", "error"]:
                timestamp = job_data.get("timestamp")
                if timestamp and timestamp < cutoff_time:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self._jobs[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old translation jobs")

    def force_cleanup(self) -> int:
        """Force immediate cleanup. Returns number of jobs removed."""
        with self._lock:
            initial_count = len(self._jobs)
            self._cleanup_old_jobs()
            return initial_count - len(self._jobs)


class StateManager:
    """Manages request-scoped state instances."""

    def __init__(self):
        self._states: Dict[str, AdvancedTranslationState] = {}
        self._lock = threading.Lock()

    def get_state(self, session_id: str) -> AdvancedTranslationState:
        """Get or create state for a session."""
        with self._lock:
            if session_id not in self._states:
                self._states[session_id] = AdvancedTranslationState()
            return self._states[session_id]

    def remove_state(self, session_id: str) -> None:
        """Remove state for a session."""
        with self._lock:
            self._states.pop(session_id, None)

    @contextmanager
    def session_state(self, session_id: str):
        """Context manager for session state."""
        state = self.get_state(session_id)
        try:
            yield state
        finally:
            # Optionally clean up state after use
            pass


# Thread-safe job manager instance
translation_jobs = ThreadSafeTranslationJobs(retention_hours=24)

# State manager for request-scoped states
state_manager = StateManager()

# For backward compatibility - should be replaced with request-scoped state
# WARNING: This global state is not thread-safe and should be migrated
state = AdvancedTranslationState()


def get_request_state(session_id: Optional[str] = None) -> AdvancedTranslationState:
    """Get state for current request/session.

    Args:
        session_id: Optional session identifier. If None, returns global state.

    Returns:
        State instance for the session

    Note:
        In production, session_id should always be provided to ensure
        thread-safety. The global state fallback is only for backward
        compatibility.
    """
    if session_id:
        return state_manager.get_state(session_id)
    else:
        # WARNING: Using global state - not thread-safe
        logger.warning(
            "Using global state instance - this is not thread-safe. "
            "Please provide a session_id for proper state isolation."
        )
        return state
