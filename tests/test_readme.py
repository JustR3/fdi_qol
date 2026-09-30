import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_cli_flags_exist():
    readme = (ROOT / "README.md").read_text()
    used = set(re.findall(r"fdi_qol\.py[^\n]*?(--[a-z-]+)", readme))
    help_text = subprocess.run([sys.executable, str(ROOT / "fdi_qol.py"), "--help"],
                               capture_output=True, text=True, check=True).stdout
    assert used and all(flag in help_text for flag in used)
