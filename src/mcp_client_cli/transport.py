"""Transport abstraction for MCP client CLI."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Union, Optional

from mcp import StdioServerParameters


@dataclass
class SseServerParameters:
    """Parameters for connecting to an SSE-based MCP server."""
    url: str
    headers: Optional[dict[str, str]] = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0


# Union type for all supported server parameter types
ServerParameters = Union[StdioServerParameters, SseServerParameters]


@asynccontextmanager
async def create_transport(params: ServerParameters):
    """
    Create a transport connection to an MCP server.

    Supports both STDIO (child process) and SSE (HTTP) transports.
    Returns (read_stream, write_stream) tuple for use with ClientSession.

    Args:
        params: Either StdioServerParameters or SseServerParameters

    Yields:
        tuple: (read_stream, write_stream) for MCP communication
    """
    if isinstance(params, StdioServerParameters):
        # STDIO transport - spawn child process
        from mcp.client.stdio import stdio_client
        async with stdio_client(params) as (read, write):
            yield read, write
    elif isinstance(params, SseServerParameters):
        # SSE transport - connect to HTTP endpoint
        from mcp.client.sse import sse_client
        async with sse_client(
            params.url,
            headers=params.headers,
            timeout=params.timeout,
            sse_read_timeout=params.sse_read_timeout
        ) as (read, write):
            yield read, write
    else:
        raise ValueError(f"Unsupported server parameter type: {type(params)}")
