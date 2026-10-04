"""Verification entry point.

Runs both passes, writes the three root artefacts and the plain-text summary, and exits
non-zero only when a code family fails.
"""

from __future__ import annotations

from immunotargetnet.verification.driver import main

if __name__ == "__main__":
    raise SystemExit(main())
