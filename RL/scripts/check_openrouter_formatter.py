"""Run with PYTHONPATH=. python scripts/check_openrouter_formatter.py."""
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from rllm.rewards import travel_reward as reward

class Response:
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]}

with TemporaryDirectory() as directory:
    key = Path(directory) / "key"
    key.write_text("test-key")
    env = {"AGENT_STAR_FORMATTER_URL": "https://openrouter.ai/api/v1",
           "AGENT_STAR_FORMATTER_MODEL": "nvidia/nemotron-3.5-super-vl:free",
           "AGENT_STAR_FORMATTER_API_KEY_FILE": str(key), "OPENROUTER_API_KEY": ""}
    with patch.dict(os.environ, env), patch.object(reward, "_formatter_session") as session:
        session.return_value.post.return_value = Response()
        assert reward.call_format_model([]) == []
        args, kwargs = session.return_value.post.call_args
        assert args[0] == "https://openrouter.ai/api/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        payload = kwargs["json"]
        assert payload["model"] == env["AGENT_STAR_FORMATTER_MODEL"]
        assert payload["reasoning"] == {"enabled": False}
        assert "stop_token_ids" not in payload and "chat_template_kwargs" not in payload
        key.unlink()
        session.return_value.post.reset_mock()
        try:
            reward.call_format_model([])
        except RuntimeError as exc:
            assert "requires" in str(exc)
        else:
            raise AssertionError("Missing credential must fail before HTTP")
        session.return_value.post.assert_not_called()
        os.environ["AGENT_STAR_FORMATTER_URL"] = "http://formatter.invalid/v1"
        reward.call_format_model([])
        assert session.return_value.post.call_args.kwargs["json"]["stop_token_ids"] == [248046, 248044]
print("OPENROUTER_FORMATTER_CHECK_PASSED")
