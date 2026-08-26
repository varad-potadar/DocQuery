"""
services/retry.py

Shared retry decorator for LLM API calls. services/qa_engine.py and
services/query_rewriter.py each call the Gemini API directly and both
need this, so it lives here once instead of twice.

Retries on:
  - 503/5xx server errors and 429 rate limits (google.genai.errors.APIError
    with a retriable .code)
  - transient network failures that never even reach the API -- DNS
    failure, connection refused/reset, read/connect timeout

Does NOT retry other 4xx client errors (bad API key, malformed request,
etc.) -- those won't be fixed by trying again, and retrying would only
delay a failure that should surface immediately.

Built on tenacity, which is already an installed dependency of
google-genai itself (see its own requirements) -- this adds no new
package to requirements.txt, just makes the existing transitive
dependency an explicit, direct one.
"""

import httpx
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# 429 (rate limit) and the 5xx family (server-side/transient) are worth
# retrying. Any other 4xx means the request itself is wrong (bad key,
# bad payload) -- retrying changes nothing about that.
RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Failures where the request never got a response at all -- these are
# the "transient network error" case, distinct from the API actively
# returning an error status.
RETRIABLE_NETWORK_ERRORS = (httpx.TransportError, TimeoutError, ConnectionError)


def _is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) in RETRIABLE_STATUS_CODES
    return isinstance(exc, RETRIABLE_NETWORK_ERRORS)


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    wait = getattr(retry_state.next_action, "sleep", 0)
    print(f"[retry] {retry_state.fn.__name__} attempt {retry_state.attempt_number} "
          f"failed ({exc}); retrying in {wait:.0f}s")


def retry_with_backoff(max_attempts: int = 5, base_delay: float = 1.0, max_delay: float = 16.0):
    """
    5 attempts by default, waiting base_delay * 2**(attempt-1) between
    them (1s, 2s, 4s, 8s), capped at max_delay. Logs every retry via
    print() (matching this codebase's existing [tag]-prefixed logging
    convention, e.g. [qa_engine]/[ingest], rather than introducing the
    stdlib logging module for just this one thing).

    reraise=True re-raises the *original* exception unchanged once
    attempts run out (or immediately, if the error isn't retriable at
    all) -- without it, tenacity would wrap the failure in its own
    RetryError, which the callers here don't expect and don't handle.
    This decorator only ever buys extra attempts; it never hides a real
    failure or changes what the caller sees when one occurs.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
        retry=retry_if_exception(_is_retriable),
        before_sleep=_log_retry,
        reraise=True,
    )
