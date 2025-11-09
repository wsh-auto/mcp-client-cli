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


@dataclass
class StreamableHttpServerParameters:
    """Parameters for connecting to a Streamable HTTP MCP server."""
    url: str
    headers: Optional[dict[str, str]] = None
    timeout: float = 30.0
    sse_read_timeout: float = 300.0


# Union type for all supported server parameter types
ServerParameters = Union[StdioServerParameters, SseServerParameters, StreamableHttpServerParameters]


@asynccontextmanager
async def create_transport(params: ServerParameters, debug: bool = False):
    """
    Create a transport connection to an MCP server.

    Supports STDIO (child process), SSE (HTTP), and Streamable HTTP transports.
    Returns (read_stream, write_stream) tuple for use with ClientSession.

    Args:
        params: Either StdioServerParameters, SseServerParameters, or StreamableHttpServerParameters
        debug: If True, server will show debug logs. If False, server logs are suppressed.

    Yields:
        tuple: (read_stream, write_stream) for MCP communication
    """
    if isinstance(params, StdioServerParameters):
        # STDIO transport - spawn child process
        # Set MCP_DEBUG environment variable to control server logging
        if not debug:
            # Add MCP_QUIET=1 to suppress server logs
            import copy
            params = copy.copy(params)
            if params.env is None:
                params.env = {}
            else:
                params.env = dict(params.env)  # Copy to avoid mutating original
            params.env['MCP_QUIET'] = '1'

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
    elif isinstance(params, StreamableHttpServerParameters):
        # Streamable HTTP transport - POST with SSE responses
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(
            params.url,
            headers=params.headers,
            timeout=params.timeout,
            sse_read_timeout=params.sse_read_timeout
        ) as (read, write, get_session_id):
            yield read, write
    else:
        raise ValueError(f"Unsupported server parameter type: {type(params)}")
