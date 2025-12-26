# Auto-inject .env from template if missing (1Password integration)
import os
import sys
import subprocess
from pathlib import Path

# Navigate from mcp_client_cli/__init__.py to query-llm/ root (5 levels up)
# mcp_client_cli/__init__.py -> mcp_client_cli/ -> src/ -> mcp-client-cli/ -> src/ -> query-llm/
_root = Path(__file__).parent.parent.parent.parent.parent
_env_path = _root / '.env'
_template_path = _root / '.env.template'

if not _env_path.exists() and _template_path.exists() and not os.environ.get('LITELLM_API_KEY'):
    print('[env] .env missing, running op inject...', file=sys.stderr)
    subprocess.run(['op', 'inject', '-i', str(_template_path), '-o', str(_env_path)], check=True)

from dotenv import load_dotenv
load_dotenv(_env_path)
