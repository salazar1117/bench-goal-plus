"""Bench-side network isolation for blind evaluation runs.

Blocks all outbound network access for the current process tree except
loopback and the configured LLM API endpoint. This prevents workers from
querying upstream diffs, CVE databases, or other external sources during
blind evaluation, which would contaminate results.

Usage:
    from bench_goal_plus.network_isolation import NetworkIsolation

    iso = NetworkIsolation(allow_endpoint="http://api.example.com:8080/v1")
    iso.apply()      # before experiment
    ...
    iso.cleanup()    # after experiment (also called on exit)
"""

from __future__ import annotations

import atexit
import ipaddress
import os
import signal
import socket
import subprocess
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class NetworkIsolation:
    """Manage iptables-based network isolation for the experiment process tree.

    Blocks all outbound traffic except loopback and the resolved IPs of the
    allowed LLM API endpoint. Uses iptables with a custom chain name so
    cleanup is idempotent and safe to call multiple times.
    """

    allow_endpoint: str
    _chain: str = "BENCH_NET_ISOLATE"
    _applied: bool = False
    _resolved_ips: list[str] = field(default_factory=list)

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["iptables", "-w", *args],
            capture_output=True,
            text=True,
        )

    def _resolve(self, hostname: str) -> list[str]:
        try:
            results = socket.getaddrinfo(hostname, None, socket.AF_INET)
            return list({r[4][0] for r in results})
        except socket.gaierror:
            return []

    def apply(self) -> bool:
        """Apply network isolation rules. Returns True on success."""
        if self._applied:
            return True
        if os.geteuid() != 0:
            # iptables requires root; fall back to no-op with a warning.
            print(
                "WARNING: network isolation requires root (iptables); "
                "skipping. Run with sudo or set DETECT_NETWORK_ISOLATION=0.",
                flush=True,
            )
            return False

        parsed = urlparse(self.allow_endpoint)
        hostname = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        self._resolved_ips = self._resolve(hostname)
        if not self._resolved_ips:
            print(
                f"WARNING: could not resolve {hostname}; "
                "network isolation may block the API endpoint.",
                flush=True,
            )

        # Clean any stale rules first
        self.cleanup()

        # Create our chain
        self._run("-N", self._chain)

        # Allow established/related connections (API responses)
        self._run("-A", self._chain, "-m", "conntrack",
                  "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT")

        # Allow loopback
        self._run("-A", self._chain, "-o", "lo", "-j", "ACCEPT")

        # Allow DNS (needed for API hostname resolution at startup)
        self._run("-A", self._chain, "-p", "udp", "--dport", "53", "-j", "ACCEPT")
        self._run("-A", self._chain, "-p", "tcp", "--dport", "53", "-j", "ACCEPT")

        # Allow the resolved API IPs
        for ip in self._resolved_ips:
            self._run("-A", self._chain, "-d", ip, "-p", "tcp",
                      "--dport", str(port), "-j", "ACCEPT")

        # Block everything else
        self._run("-A", self._chain, "-j", "DROP")

        # Insert our chain at the top of OUTPUT
        self._run("-I", "OUTPUT", "1", "-j", self._chain)

        self._applied = True
        atexit.register(self.cleanup)
        print(
            f"Network isolation applied: outbound blocked except "
            f"loopback, DNS, and {hostname}:{port} ({', '.join(self._resolved_ips) or 'unresolved'})",
            flush=True,
        )
        return True

    def cleanup(self) -> None:
        """Remove network isolation rules. Safe to call multiple times."""
        if os.geteuid() != 0:
            return
        # Remove jump rule from OUTPUT
        while self._run("-C", "OUTPUT", "-j", self._chain).returncode == 0:
            self._run("-D", "OUTPUT", "-j", self._chain)
        # Flush and delete the chain
        self._run("-F", self._chain)
        self._run("-X", self._chain)
        self._applied = False
