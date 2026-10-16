# netrun

`/etc/resolv.conf` on these nodes points at a stub that answers nothing, so any
outbound name lookup fails. `netrun.sh` bind-mounts a resolver that works over it for
one command, inside an unprivileged user namespace. Nothing on the node changes.

    workflows/netrun/netrun.sh <command> [args...]

The replacement resolver is read from beside the script, so moving or copying this
directory keeps it working. `NETRUN_RESOLV` overrides the path.

This lived outside the repository until 2026-10-06, under a home directory, and named
that path directly. A reservation covering the filesystem that home sits on made the
cost of that clear: a copy taken to another filesystem still pointed back at the one
it was copied to escape, and without the wrapper nothing can be downloaded at all.

`resolv.conf` here names only public resolvers.

Whether the stub is still in place on every node is not established -- a probe on
c167 on 2026-10-06 resolved both with and without the wrapper. The wrapper is kept
because the jobs that need the network cannot choose their node.
