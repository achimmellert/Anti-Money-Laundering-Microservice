"""
AML Test Inference Streamer (Alias for send_test_transactions.py)
=================================================================
Allows running inference simulation using `send_test_inference.py` or
`send_test_transactions.py` interchangeably.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.send_test_transactions import main

if __name__ == "__main__":
    main()
