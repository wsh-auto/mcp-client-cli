# Auto-inject .env from template if missing (1Password integration)
import os
import sys
import subprocess
from pathlib import Path

# Navigate from mcp_client_cli/__init__.py to mcp-client-cli/ root
_root = Path(__file__).parent.parent.parent
_env_path = _root / '.env'
_template_path = _root / '.env.template'

if not _env_path.exists() and _template_path.exists():
    print('[env] .env missing, running op inject...', file=sys.stderr)
    subprocess.run(['op', 'inject', '-i', str(_template_path), '-o', str(_env_path)], check=True)

from dotenv import load_dotenv
load_dotenv(_env_path)
