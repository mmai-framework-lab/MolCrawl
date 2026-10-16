#!/bin/bash
# Run a command with a working DNS resolver.
#
# This cluster's /etc/resolv.conf points at a stub that answers nothing, so any
# outbound name lookup fails. An unprivileged user namespace lets a resolv.conf that
# does work be bind-mounted over it for one command, without changing the node.
#
# The replacement is read from beside this script rather than from an absolute path,
# so a copy of this directory works wherever it is placed. The original lived under
# /home and named it directly, which would have made a copy on another filesystem
# point back at the one it was copied to escape.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLV="${NETRUN_RESOLV:-${HERE}/resolv.conf}"
[ -r "${RESOLV}" ] || { echo "netrun: no readable resolv.conf at ${RESOLV}" >&2; exit 1; }
exec unshare --map-root-user --mount -- bash -c '
  mount --bind "$1" /etc/resolv.conf || exit 1
  shift
  exec "$@"
' _ "${RESOLV}" "$@"
