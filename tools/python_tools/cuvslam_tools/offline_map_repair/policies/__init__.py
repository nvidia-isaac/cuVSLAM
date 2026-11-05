"""Client-side repair policies: what to change in a map, and why.

Each module is one self-contained experiment that reads a cached map, decides on an
edit, and commits it to a fresh clone through MCP. The server never sees the policy;
it only applies opaque, compare-and-swap record changes.
"""
