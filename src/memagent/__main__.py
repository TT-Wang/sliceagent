"""Enable `python -m memagent`. The console-script (`memagent`) regenerates with a stale shebang when the
venv is moved, so this module path is the reliable way to launch from a source checkout."""
from .cli import main

if __name__ == "__main__":
    main()
