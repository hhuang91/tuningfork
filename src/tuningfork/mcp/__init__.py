"""tuningfork's operations as tools for an agent, over the Model Context Protocol.

The same operations the CLI and the web UI use (`tuningfork.ops`), for an agent: Claude
Code, Hermes, an OpenJarvis teacher. Reads run here, on the recordings; work that calls the
model runs as jobs inside the proxy, through its gate, like work started from the web UI.

`tuningfork mcp` serves them over stdio; a running proxy serves them over Streamable HTTP
too. Only those two import this package: the MCP SDK is an optional extra
(`tuningfork-lab[mcp]`), and nothing else in tuningfork needs it.
"""
