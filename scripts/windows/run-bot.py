import signal
import sys
from pathlib import Path

from nezubot_discrod.bot import main
from nezubot_discrod.sandbox import CONTAINER_LABEL, shutdown_sandboxes

signal.signal(signal.SIGINT, signal.default_int_handler)
Path(sys.argv[1]).write_text(CONTAINER_LABEL, encoding="ascii")
try:
    main()
finally:
    shutdown_sandboxes(timeout=5)
