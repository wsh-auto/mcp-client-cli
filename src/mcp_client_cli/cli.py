#!/usr/bin/env python3

"""
Simple lll CLI that acts as MCP client.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datetime import datetime
import argparse
import asyncio
from typing import Annotated, TypedDict
import uuid
import sys
import re
import anyio
import time

# Capture time as early as possible (after basic imports, before heavy ones)
_MODULE_LOAD_START = time.time()

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langgraph.managed import IsLastStep
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.table import Table
import base64
import imghdr as imghdr
import mimetypes

from .input import *
from .const import *
from .output import *
from .storage import *
from .tool import *
from .prompt import *
from .memory import *
from .config import AppConfig

# The AgentState class is used to maintain the state of the agent during a conversation.
class AgentState(TypedDict):
    # A list of messages exchanged in the conversation.
    messages: Annotated[list[BaseMessage], add_messages]
    # A flag indicating whether the current step is the last step in the conversation.
    is_last_step: IsLastStep
    # The current date and time, used for context in the conversation.
    today_datetime: str
    # The user's memories.
    memories: str = "no memories"
    remaining_steps: int = 5

async def run(total_start_time: float) -> None:
    """Run the LLM agent."""
    args = setup_argument_parser()

    # Handle --help or no arguments: show help + models
    if args.help or len(sys.argv) == 1:
        # Create parser to print help first
        parser = create_parser()
        parser.print_help()
        print()  # Add spacing before tables
        # Load config to display model information
        app_config = AppConfig.load(args.config)
        handle_list_models(app_config)
        return

    query, is_conversation_continuation = parse_query(args)
    app_config = AppConfig.load(args.config)

    if args.list_tools:
        await handle_list_tools(app_config, args)
        return

    if args.show_memories:
        await handle_show_memories()
        return

    if args.list_prompts:
        handle_list_prompts()
        return

    await handle_conversation(args, query, is_conversation_continuation, app_config, total_start_time)

def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser without parsing."""
    parser = argparse.ArgumentParser(
        prog='lll',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # Disable default -h/--help to add custom handler
        epilog="""
Examples:
  lll "What is the capital of France?"     Ask a simple question
  lll c "tell me more"                     Continue previous conversation
  lll p review                             Use a prompt template
  cat file.txt | lll                       Process input from a file
  lll --list-tools                         Show available tools
  lll --list-prompts                       Show available prompt templates
  lll --no-confirmations "search web"      Run tools without confirmation

AGENTS: MUST load skill://mdr:test-mcp-server before editing or for context
        """
    )
    parser.add_argument('query', nargs='*', default=[],
                       help=argparse.SUPPRESS)
    parser.add_argument('-h', '--help', action='store_true',
                       help='Show this help message with model information and exit')
    parser.add_argument('--list-tools', action='store_true',
                       help='List all available LLM tools')
    parser.add_argument('--list-prompts', action='store_true',
                       help='List all available prompts')
    parser.add_argument('--no-confirmations', action='store_true',
                       help='Bypass tool confirmation requirements')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of tools capabilities')
    parser.add_argument('--text-only', action='store_true',
                       help='Print output as raw text instead of parsing markdown')
    parser.add_argument('--mcp', action='store_true',
                       help='Enable MCP tools (slower startup, more capable)')
    parser.add_argument('--no-intermediates', action='store_true',
                       help='Only print the final message')
    parser.add_argument('--show-memories', action='store_true',
                       help='Show user memories')
    parser.add_argument('--model',
                       help='Override the model specified in config')
    parser.add_argument('--config',
                       help='Path to config file (default: ~/.lll/config.json)')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information including server logs and timing')
    return parser

def setup_argument_parser() -> argparse.Namespace:
    """Setup and parse arguments."""
    return create_parser().parse_args()

def handle_list_models(app_config: AppConfig) -> None:
    """Handle the --list-models command."""
    console = Console()

    llm_config = app_config.llm

    config_table = Table()
    config_table.add_column("Property", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="green")

    config_table.add_row("Provider", llm_config.provider or "Not specified")
    config_table.add_row("Model", llm_config.model or "Not specified")
    config_table.add_row("Base URL", llm_config.base_url or "Default")
    config_table.add_row("API Key", "***" + (llm_config.api_key[-8:] if llm_config.api_key and len(llm_config.api_key) > 8 else "Set") if llm_config.api_key else "Not set")

    console.print(config_table)
    print()

    # Show all available LiteLLM models
    models_table = Table()
    models_table.add_column("Model", style="cyan", no_wrap=True)
    models_table.add_column("Context", style="yellow", justify="right")
    models_table.add_column("Input", style="green", justify="right")
    models_table.add_column("Output", style="green", justify="right")
    models_table.add_column("Throughput", style="magenta", justify="right")
    models_table.add_column("TTFT", style="bright_cyan", justify="right")
    models_table.add_column("TTLT", style="bright_blue", justify="right")

    # Model pricing and performance from OpenRouter (Nov 2025)
    # Sorted by TTFT (Time to First Token, lowest first)
    models = [
        ("google/gemini-2.5-flash-lite", "1M", "$0.10", "$0.40", "78 tok/s", "0.36s", "0.76s"),
        ("google/gemini-2.5-flash", "1M", "$0.30", "$2.50", "94 tok/s", "0.47s", "0.77s"),
        ("anthropic/claude-haiku-4.5", "200K", "$1.00", "$5.00", "138 tok/s", "0.51s", "1.04s"),
        ("anthropic/claude-sonnet-4.5", "1M", "$3.00", "$15.00", "62 tok/s", "1.27s", "1.29s"),
        ("x-ai/grok-4-fast", "2M", "$0.20", "$0.50", "140 tok/s", "3.72s", "10.19s"),
        ("openai/gpt-5", "400K", "$1.25", "$10.00", "67 tok/s", "7.31s", "7.82s"),
        ("x-ai/grok-4", "256K", "$3.00", "$15.00", "33 tok/s", "15.59s", "45.09s"),
    ]

    for model, context, input_price, output_price, throughput, ttft, ttlt in models:
        models_table.add_row(model, context, input_price, output_price, throughput, ttft, ttlt)

    console.print(models_table)

async def handle_list_tools(app_config: AppConfig, args: argparse.Namespace) -> None:
    """Handle the --list-tools command."""
    server_configs = [
        McpServerConfig(
            server_name=name,
            server_param=config.to_transport_params(),
            exclude_tools=config.exclude_tools or []
        )
        for name, config in app_config.get_enabled_servers().items()
    ]
    toolkits, tools = await load_tools(server_configs, args.mcp, args.force_refresh, args.debug)

    console = Console()
    table = Table(title="Available LLM Tools")
    table.add_column("Toolkit", style="cyan")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="green")

    for tool in tools:
        if isinstance(tool, McpTool):
            table.add_row(tool.toolkit_name, tool.name, tool.description)

    console.print(table)

    for toolkit in toolkits:
        await toolkit.close()

async def handle_show_memories() -> None:
    """Handle the --show-memories command."""
    store = SqliteStore(SQLITE_DB)
    memories = await get_memories(store)
    console = Console()
    table = Table(title="My LLM Memories")
    for memory in memories:
        table.add_row(memory)
    console.print(table)

def handle_list_prompts() -> None:
    """Handle the --list-prompts command."""
    console = Console()
    table = Table(title="Available Prompt Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Template")
    table.add_column("Arguments")

    for name, template in prompt_templates.items():
        table.add_row(name, template, ", ".join(re.findall(r'\{(\w+)\}', template)))

    console.print(table)

def show_model_error_and_list() -> None:
    """Show available models when there's a model error."""
    console = Console()

    models_table = Table(title="Available LiteLLM Models")
    models_table.add_column("Model", style="cyan", no_wrap=True)
    models_table.add_column("Context", style="yellow", justify="right")
    models_table.add_column("Input", style="green", justify="right")
    models_table.add_column("Output", style="green", justify="right")
    models_table.add_column("Throughput", style="magenta", justify="right")
    models_table.add_column("TTFT", style="bright_cyan", justify="right")
    models_table.add_column("TTLT", style="bright_blue", justify="right")

    # Model pricing and performance from OpenRouter (Nov 2025)
    # Sorted by TTFT (Time to First Token, lowest first)
    models = [
        ("google/gemini-2.5-flash-lite", "1M", "$0.10", "$0.40", "78 tok/s", "0.36s", "0.76s"),
        ("google/gemini-2.5-flash", "1M", "$0.30", "$2.50", "94 tok/s", "0.47s", "0.77s"),
        ("anthropic/claude-haiku-4.5", "200K", "$1.00", "$5.00", "138 tok/s", "0.51s", "1.04s"),
        ("anthropic/claude-sonnet-4.5", "1M", "$3.00", "$15.00", "62 tok/s", "1.27s", "1.29s"),
        ("x-ai/grok-4-fast", "2M", "$0.20", "$0.50", "140 tok/s", "3.72s", "10.19s"),
        ("openai/gpt-5", "400K", "$1.25", "$10.00", "67 tok/s", "7.31s", "7.82s"),
        ("x-ai/grok-4", "256K", "$3.00", "$15.00", "33 tok/s", "15.59s", "45.09s"),
    ]

    for model, context, input_price, output_price, throughput, ttft, ttlt in models:
        models_table.add_row(model, context, input_price, output_price, throughput, ttft, ttlt)

    console.print(models_table)
    print("\nTip: Use 'lll --list-models' to see configured model and available options")
    print()

async def load_tools(server_configs: list[McpServerConfig], enable_mcp: bool, force_refresh: bool, debug: bool = False) -> tuple[list, list]:
    """Load and convert MCP tools to LangChain tools."""
    if not enable_mcp:
        return [], []

    toolkits = []
    langchain_tools = []

    async def convert_toolkit(server_config: McpServerConfig):
        toolkit = await convert_mcp_to_langchain_tools(server_config, force_refresh, debug)
        toolkits.append(toolkit)
        langchain_tools.extend(toolkit.get_tools())

    async with anyio.create_task_group() as tg:
        for server_param in server_configs:
            tg.start_soon(convert_toolkit, server_param)

    langchain_tools.append(save_memory)
    return toolkits, langchain_tools

async def handle_simple_conversation(model, query: HumanMessage, app_config: AppConfig,
                                   args: argparse.Namespace, total_start_time: float) -> None:
    """Handle simple conversation without MCP tools (fast path)."""
    # Track timing
    start_time = time.time()
    first_token_time = None
    last_token_time = None

    try:
        # Simple streaming without agent infrastructure
        messages = [
            {"role": "system", "content": app_config.system_prompt},
            {"role": "user", "content": query.content}
        ]

        async for chunk in model.astream(messages):
            # Record first and last token times
            if chunk.content:
                if first_token_time is None:
                    first_token_time = time.time()
                last_token_time = time.time()

            # Output the chunk (simple text output)
            if chunk.content:
                print(chunk.content, end="", flush=True)

    except Exception as e:
        # Check if this is a model-related error
        error_str = str(e).lower()
        if "invalid model" in error_str or "model not found" in error_str or "model=" in error_str:
            print(f"\n❌ Invalid model '{app_config.llm.model}':")
            print(f"   {str(e)}\n")
            show_model_error_and_list()
        else:
            print(f"\n❌ Error: {str(e)}", file=sys.stderr)
    finally:
        print()  # Add newline after response

        # Show timing information if we got a response
        if first_token_time is not None:
            ttft = first_token_time - start_time
            total_time = time.time() - total_start_time

            # Format model name (truncate if too long)
            model_name = app_config.llm.model
            if len(model_name) > 40:
                model_name = model_name[:37] + "..."

            # ANSI color codes for dimmed/gray text (debug styling)
            GRAY = '\033[90m'
            RESET = '\033[0m'

            # Add newline before timing output
            print(file=sys.stderr)
            print(f"{GRAY}⏱️  [{model_name}] TTFT: {ttft:.2f}s", file=sys.stderr, end="")

            if last_token_time is not None:
                ttlt = last_token_time - start_time
                print(f"  |  TTLT: {ttlt:.2f}s", file=sys.stderr, end="")

            # Show total time including loading
            print(f"  |  Total: {total_time:.2f}s", file=sys.stderr, end="")
            print(RESET, file=sys.stderr)  # Reset color and new line at end

async def handle_conversation(args: argparse.Namespace, query: HumanMessage,
                            is_conversation_continuation: bool, app_config: AppConfig,
                            total_start_time: float) -> None:
    """Handle the main conversation flow."""
    server_configs = [
        McpServerConfig(
            server_name=name,
            server_param=config.to_transport_params(),
            exclude_tools=config.exclude_tools or []
        )
        for name, config in app_config.get_enabled_servers().items()
    ]
    toolkits, tools = await load_tools(server_configs, args.mcp, args.force_refresh, args.debug)

    extra_body = {}
    if app_config.llm.base_url and "openrouter" in app_config.llm.base_url:
        extra_body = {"transforms": ["middle-out"]}
    # Override model if specified in command line
    if args.model:
        app_config.llm.model = args.model

    # Build init_chat_model kwargs, only include provider if specified
    init_kwargs = {
        "model": app_config.llm.model,
        "api_key": app_config.llm.api_key,
        "temperature": app_config.llm.temperature,
        "base_url": app_config.llm.base_url,
        "default_headers": {
            "X-Title": "mcp-client-cli",
            "HTTP-Referer": "https://github.com/adhikasp/mcp-client-cli",
        },
        "extra_body": extra_body
    }
    if app_config.llm.provider:
        init_kwargs["model_provider"] = app_config.llm.provider

    # Try to initialize the model, catch errors and show helpful message
    try:
        model: BaseChatModel = init_chat_model(**init_kwargs)
    except Exception as e:
        print(f"\n❌ Error initializing model '{app_config.llm.model}':")
        print(f"   {str(e)}\n")
        show_model_error_and_list()
        for toolkit in toolkits:
            await toolkit.close()
        return

    # Simple direct LLM path when MCP is disabled (much faster startup)
    if not args.mcp:
        await handle_simple_conversation(model, query, app_config, args, total_start_time)
        for toolkit in toolkits:
            await toolkit.close()
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.system_prompt),
        ("placeholder", "{messages}")
    ])

    conversation_manager = ConversationManager(SQLITE_DB)
    
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB) as checkpointer:
        store = SqliteStore(SQLITE_DB)
        memories = await get_memories(store)
        formatted_memories = "\n".join(f"- {memory}" for memory in memories)
        agent_executor = create_react_agent(
            model, tools, state_schema=AgentState,
            prompt=prompt, checkpointer=checkpointer, store=store
        )
        
        thread_id = (await conversation_manager.get_last_id() if is_conversation_continuation 
                    else uuid.uuid4().hex)

        input_messages = AgentState(
            messages=[query], 
            today_datetime=datetime.now().isoformat(),
            memories=formatted_memories,
            remaining_steps=3
        )

        output = OutputHandler(text_only=args.text_only, only_last_message=args.no_intermediates)
        output.start()

        # Track timing for TTFT (Time To First Token) and TTLT (Time To Last Token)
        start_time = time.time()
        first_token_time = None
        last_token_time = None

        try:
            async for chunk in agent_executor.astream(
                input_messages,
                stream_mode=["messages", "values"],
                config={"configurable": {"thread_id": thread_id, "user_id": "myself"},
                       "recursion_limit": 100}
            ):
                # Record first and last token times
                from langchain_core.messages import AIMessageChunk
                if isinstance(chunk, tuple) and chunk[0] == "messages":
                    message_chunk = chunk[1][0] if len(chunk[1]) > 0 else None
                    if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        last_token_time = time.time()

                output.update(chunk)
                if not args.no_confirmations:
                    if not output.confirm_tool_call(app_config.__dict__, chunk):
                        break
        except Exception as e:
            # Check if this is a model-related error
            error_str = str(e).lower()
            if "invalid model" in error_str or "model not found" in error_str or "model=" in error_str:
                output.finish()
                print(f"\n❌ Invalid model '{app_config.llm.model}':")
                print(f"   {str(e)}\n")
                show_model_error_and_list()
            else:
                output.update_error(e)
        finally:
            output.finish()

            # Show timing information if we got a response
            if first_token_time is not None:
                ttft = first_token_time - start_time
                total_time = time.time() - total_start_time

                # Format model name (truncate if too long)
                model_name = app_config.llm.model
                if len(model_name) > 40:
                    model_name = model_name[:37] + "..."

                # ANSI color codes for dimmed/gray text (debug styling)
                DIM = '\033[2m'      # Dim/faint text
                GRAY = '\033[90m'    # Bright black (gray)
                RESET = '\033[0m'

                # Add newline before timing output
                print(file=sys.stderr)
                print(f"{GRAY}⏱️  [{model_name}] TTFT: {ttft:.2f}s", file=sys.stderr, end="")

                if last_token_time is not None:
                    ttlt = last_token_time - start_time
                    print(f"  |  TTLT: {ttlt:.2f}s", file=sys.stderr, end="")

                # Show total time including loading
                print(f"  |  Total: {total_time:.2f}s", file=sys.stderr, end="")

                # Detect thinking/reasoning models
                thinking_models = ["o1", "o3", "deepseek-r1", "qwen-qwq"]
                is_thinking = any(tm in model_name.lower() for tm in thinking_models)

                if is_thinking:
                    print(f"  |  [THINKING]", file=sys.stderr, end="")

                print(RESET, file=sys.stderr)  # Reset color and new line at end

        await conversation_manager.save_id(thread_id, checkpointer.conn)

    for toolkit in toolkits:
        await toolkit.close()

def parse_query(args: argparse.Namespace) -> tuple[HumanMessage, bool]:
    """
    Parse the query from command line arguments.
    Returns a tuple of (HumanMessage, is_conversation_continuation).
    """
    query_parts = ' '.join(args.query).split()
    stdin_content = ""
    stdin_image = None
    is_continuation = False

    # Handle clipboard content if requested
    if query_parts and query_parts[0] == 'cb':
        # Remove 'cb' from query parts
        query_parts = query_parts[1:]
        # Try to get content from clipboard
        clipboard_result = get_clipboard_content()
        if clipboard_result:
            content, mime_type = clipboard_result
            if mime_type:  # It's an image
                stdin_image = base64.b64encode(content).decode('utf-8')
            else:  # It's text
                stdin_content = content
        else:
            print("No content found in clipboard")
            raise Exception("Clipboard is empty")
    # Check if there's input from pipe
    elif not sys.stdin.isatty():
        stdin_data = sys.stdin.buffer.read()
        # Try to detect if it's an image
        image_type = imghdr.what(None, h=stdin_data)
        if image_type:
            # It's an image, encode it as base64
            stdin_image = base64.b64encode(stdin_data).decode('utf-8')
            mime_type = mimetypes.guess_type(f"dummy.{image_type}")[0] or f"image/{image_type}"
        else:
            # It's text
            stdin_content = stdin_data.decode('utf-8').strip()

    # Process the query text
    query_text = ""
    if query_parts:
        if query_parts[0] == 'c':
            is_continuation = True
            query_text = ' '.join(query_parts[1:])
        elif query_parts[0] == 'p' and len(query_parts) >= 2:
            template_name = query_parts[1]
            if template_name not in prompt_templates:
                print(f"Error: Prompt template '{template_name}' not found.")
                print("Available templates:", ", ".join(prompt_templates.keys()))
                return HumanMessage(content=""), False

            template = prompt_templates[template_name]
            template_args = query_parts[2:]
            try:
                # Extract variable names from the template
                var_names = re.findall(r'\{(\w+)\}', template)
                # Create dict mapping parameter names to arguments
                template_vars = dict(zip(var_names, template_args))
                query_text = template.format(**template_vars)
            except KeyError as e:
                print(f"Error: Missing argument {e}")
                return HumanMessage(content=""), False
        else:
            query_text = ' '.join(query_parts)

    # Combine stdin content with query text if both exist
    if stdin_content and query_text:
        query_text = f"{stdin_content}\n\n{query_text}"
    elif stdin_content:
        query_text = stdin_content
    elif not query_text and not stdin_image:
        return HumanMessage(content=""), False

    # Create the message content
    if stdin_image:
        content = [
            {"type": "text", "text": query_text or "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{stdin_image}"}}
        ]
    else:
        content = query_text

    return HumanMessage(content=content), is_continuation

def main() -> None:
    """Entry point of the script."""
    # Use the module load time captured before heavy imports
    asyncio.run(run(_MODULE_LOAD_START))


if __name__ == "__main__":
    main()
