import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from rllm.rewards.travel_reward import _wait_formatter_rate_limit

with TemporaryDirectory() as directory:
    with patch.dict(os.environ, {
        "AGENT_STAR_FORMATTER_RPM":"600",
        "AGENT_STAR_FORMATTER_RATE_FILE":str(Path(directory)/"rate")
    }):
        start=time.monotonic()
        for _ in range(3):
            _wait_formatter_rate_limit()
        assert time.monotonic()-start >= .19
print("FORMATTER_PACING_CHECK_PASSED")
