"""Unit tests for segmentation_service.client.cli.

No real HTTP calls are made; httpx.Client is patched at the module level so
the SegmentationClient and the main() entry point can be exercised without a
running server.

The tests deliberately avoid importing anything from the adapter layer — the
client must be decoupled from backend internals.
"""

from __future__ import annotations

import argparse
import base64
import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from segmentation_service.client.cli import (
    SegmentationClient,
    _build_prompt_payload,
    _build_synthetic_prompt,
    load_image_b64,
    main,
    select_prompt,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# 1×1 PNG used wherever a real image is needed.
_STUB_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_CAPS_MOCK = {
    "backend": "mock",
    "supported_prompt_types": ["point", "box", "text"],
    "supported_input_types": ["base64", "url"],
    "max_image_width": 4096,
    "max_image_height": 4096,
}
_CAPS_SAM2 = {
    "backend": "sam2",
    "supported_prompt_types": ["point", "box"],
    "supported_input_types": ["base64", "url"],
    "max_image_width": 4096,
    "max_image_height": 4096,
}
_CAPS_CLIPSEG = {
    "backend": "clipseg",
    "supported_prompt_types": ["text"],
    "supported_input_types": ["base64", "url"],
    "max_image_width": 4096,
    "max_image_height": 4096,
}

_SEGMENT_RESPONSE = {
    "request_id": "abc-123",
    "backend": "mock",
    "masks": [
        {"mask_b64": _STUB_IMAGE_B64, "score": 0.95, "area": 1, "logits_b64": None}
    ],
    "latency_ms": 5.0,
    "metadata": {},
}


# ---------------------------------------------------------------------------
# Helpers for building mock httpx responses
# ---------------------------------------------------------------------------

def _mock_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Return a MagicMock that behaves like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _mock_http_client(responses: dict[str, Any]) -> MagicMock:
    """Return a mock httpx.Client that maps method+path to a response.

    *responses* maps ``"GET /path"`` or ``"POST /path"`` to a MagicMock response.
    """
    instance = MagicMock()

    def _get(url, **kwargs):
        path = "/" + url.split("/", 3)[-1]  # strip scheme+host
        key = f"GET {path}"
        if key in responses:
            return responses[key]
        raise KeyError(f"Unmapped GET {url}")

    def _post(url, **kwargs):
        path = "/" + url.split("/", 3)[-1]
        key = f"POST {path}"
        if key in responses:
            return responses[key]
        raise KeyError(f"Unmapped POST {url}")

    instance.get.side_effect = _get
    instance.post.side_effect = _post
    instance.__enter__.return_value = instance
    instance.__exit__.return_value = False
    return instance


# ---------------------------------------------------------------------------
# select_prompt — pure function tests
# ---------------------------------------------------------------------------

class TestSelectPrompt:
    # -- text backend ---------------------------------------------------

    def test_text_prompt_selected_when_supported(self) -> None:
        result = select_prompt(["text"], text_prompt="the cat")
        assert result == {"prompt_type": "text", "text_prompt": "the cat"}

    def test_text_prompt_ignored_when_not_supported(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            select_prompt(["point", "box"], text_prompt="the cat")

    # -- point backend --------------------------------------------------

    def test_point_prompt_selected_when_supported(self) -> None:
        result = select_prompt(["point", "box"], point="10,20,1")
        assert result["prompt_type"] == "point"
        assert result["points"] == [{"x": 10.0, "y": 20.0, "label": 1}]

    def test_point_prompt_ignored_when_not_supported(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            select_prompt(["text"], point="10,20,1")

    # -- box backend ----------------------------------------------------

    def test_box_prompt_selected_when_supported(self) -> None:
        result = select_prompt(["box"], box="0,0,100,100")
        assert result["prompt_type"] == "box"
        b = result["box"]
        assert b == {"x_min": 0.0, "y_min": 0.0, "x_max": 100.0, "y_max": 100.0}

    def test_box_prompt_ignored_when_not_supported(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            select_prompt(["text"], box="0,0,100,100")

    # -- priority: text > point > box -----------------------------------

    def test_text_wins_over_point_when_both_supported(self) -> None:
        result = select_prompt(
            ["point", "box", "text"],
            text_prompt="cat",
            point="10,20,1",
        )
        assert result["prompt_type"] == "text"

    def test_point_wins_over_box_when_text_not_supported(self) -> None:
        result = select_prompt(
            ["point", "box"],
            point="10,20,1",
            box="0,0,100,100",
        )
        assert result["prompt_type"] == "point"

    def test_box_used_as_fallback_when_only_box_supported(self) -> None:
        result = select_prompt(
            ["box"],
            text_prompt="cat",  # not supported → ignored
            point="10,20,1",    # not supported → ignored
            box="0,0,50,50",
        )
        assert result["prompt_type"] == "box"

    # -- synthetic fallback (no user prompts) ---------------------------

    def test_synthetic_text_when_no_user_prompt(self) -> None:
        result = select_prompt(["text"])
        assert result["prompt_type"] == "text"
        assert result["text_prompt"]  # non-empty placeholder

    def test_synthetic_point_when_no_user_prompt(self) -> None:
        result = select_prompt(["point", "box"])
        # text is preferred; point is first in ("text","point","box") after text
        assert result["prompt_type"] == "point"

    def test_synthetic_box_when_only_box_supported(self) -> None:
        result = select_prompt(["box"])
        assert result["prompt_type"] == "box"

    # -- parse errors ---------------------------------------------------

    def test_bad_point_string_raises(self) -> None:
        with pytest.raises(ValueError, match="x,y,label"):
            select_prompt(["point"], point="10,20")  # missing label

    def test_bad_box_string_raises(self) -> None:
        with pytest.raises(ValueError, match="xmin,ymin,xmax,ymax"):
            select_prompt(["box"], box="0,0,100")  # only 3 values


# ---------------------------------------------------------------------------
# _build_prompt_payload — edge cases
# ---------------------------------------------------------------------------

class TestBuildPromptPayload:
    def test_text(self) -> None:
        r = _build_prompt_payload("text", "a chair")
        assert r == {"prompt_type": "text", "text_prompt": "a chair"}

    def test_point_float_coords(self) -> None:
        r = _build_prompt_payload("point", "1.5,2.5,1")
        assert r["points"][0]["x"] == 1.5
        assert r["points"][0]["y"] == 2.5

    def test_box_parses_four_values(self) -> None:
        r = _build_prompt_payload("box", "10,20,200,300")
        b = r["box"]
        assert b["x_min"] == 10.0
        assert b["y_min"] == 20.0
        assert b["x_max"] == 200.0
        assert b["y_max"] == 300.0

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError):
            _build_prompt_payload("polygon", "1,2,3")


# ---------------------------------------------------------------------------
# _build_synthetic_prompt
# ---------------------------------------------------------------------------

class TestBuildSyntheticPrompt:
    def test_text_fallback_has_text_prompt(self) -> None:
        r = _build_synthetic_prompt("text")
        assert r["prompt_type"] == "text"
        assert isinstance(r["text_prompt"], str)

    def test_point_fallback_has_points(self) -> None:
        r = _build_synthetic_prompt("point")
        assert r["prompt_type"] == "point"
        assert isinstance(r["points"], list) and len(r["points"]) == 1

    def test_box_fallback_has_box(self) -> None:
        r = _build_synthetic_prompt("box")
        assert r["prompt_type"] == "box"
        assert "box" in r

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError):
            _build_synthetic_prompt("polygon")


# ---------------------------------------------------------------------------
# SegmentationClient — HTTP calls (mocked httpx.Client)
# ---------------------------------------------------------------------------

class TestSegmentationClient:
    def _make_client(self, mock_http: MagicMock) -> SegmentationClient:
        """Patch httpx.Client context manager and return a SegmentationClient."""
        with patch("segmentation_service.client.cli.httpx.Client") as mock_cls:
            mock_cls.return_value = mock_http
            client = SegmentationClient("http://test:8000")
        # Keep the mock attached for later assertions.
        client._mock_http = mock_http
        client._mock_cls = mock_cls
        return client

    def _patched(self, caps_resp, seg_resp=None):
        """Return (SegmentationClient, mock_cls) with pre-wired responses."""
        http = _mock_http_client(
            {
                "GET /api/v1/capabilities": _mock_response(caps_resp),
                "GET /api/v1/health": _mock_response({"status": "ok", "version": "0.1.0", "backend": "mock"}),
                **({"POST /api/v1/segment": _mock_response(seg_resp)} if seg_resp else {}),
            }
        )
        return http

    def test_get_capabilities_returns_dict(self) -> None:
        http = self._patched(_CAPS_MOCK)
        with patch("segmentation_service.client.cli.httpx.Client", return_value=http):
            client = SegmentationClient("http://test:8000")
            caps = client.get_capabilities()
        assert caps["backend"] == "mock"
        assert "supported_prompt_types" in caps

    def test_get_capabilities_calls_correct_path(self) -> None:
        http = self._patched(_CAPS_MOCK)
        with patch("segmentation_service.client.cli.httpx.Client", return_value=http):
            client = SegmentationClient("http://test:8000")
            client.get_capabilities()
        http.get.assert_called_once()
        url_arg = http.get.call_args.args[0]
        assert url_arg.endswith("/api/v1/capabilities")

    def test_segment_returns_dict(self) -> None:
        http = self._patched(_CAPS_MOCK, _SEGMENT_RESPONSE)
        with patch("segmentation_service.client.cli.httpx.Client", return_value=http):
            client = SegmentationClient("http://test:8000")
            result = client.segment({"image": _STUB_IMAGE_B64, "prompt_type": "text", "text_prompt": "cat"})
        assert result["backend"] == "mock"
        assert len(result["masks"]) == 1

    def test_segment_calls_correct_path(self) -> None:
        http = self._patched(_CAPS_MOCK, _SEGMENT_RESPONSE)
        with patch("segmentation_service.client.cli.httpx.Client", return_value=http):
            client = SegmentationClient("http://test:8000")
            client.segment({})
        url_arg = http.post.call_args.args[0]
        assert url_arg.endswith("/api/v1/segment")

    def test_base_url_trailing_slash_stripped(self) -> None:
        http = self._patched(_CAPS_MOCK)
        with patch("segmentation_service.client.cli.httpx.Client", return_value=http):
            client = SegmentationClient("http://test:8000/")
            client.get_capabilities()
        url_arg = http.get.call_args.args[0]
        assert "//" not in url_arg.replace("http://", "PROTO")


# ---------------------------------------------------------------------------
# main() integration — end-to-end with mocked HTTP
# ---------------------------------------------------------------------------

def _run_main_with_mocked_http(argv: list[str], caps: dict, seg: dict) -> int:
    """Run main(argv) with httpx.Client fully mocked."""
    http = _mock_http_client(
        {
            "GET /api/v1/capabilities": _mock_response(caps),
            "POST /api/v1/segment": _mock_response(seg),
        }
    )
    with patch("segmentation_service.client.cli.httpx.Client", return_value=http):
        return main(argv)


class TestMain:
    def test_text_backend_exits_zero(self, capsys) -> None:
        code = _run_main_with_mocked_http(
            ["--base-url", "http://test", "--text-prompt", "the cat"],
            _CAPS_CLIPSEG,
            {**_SEGMENT_RESPONSE, "backend": "clipseg"},
        )
        assert code == 0

    def test_point_backend_exits_zero(self, capsys) -> None:
        code = _run_main_with_mocked_http(
            ["--base-url", "http://test", "--point", "10,20,1"],
            _CAPS_SAM2,
            {**_SEGMENT_RESPONSE, "backend": "sam2"},
        )
        assert code == 0

    def test_unsupported_prompt_exits_one(self, capsys) -> None:
        """text-prompt sent to a SAM2 backend that doesn't support text."""
        code = _run_main_with_mocked_http(
            ["--base-url", "http://test", "--text-prompt", "cat"],
            _CAPS_SAM2,    # only supports point,box
            _SEGMENT_RESPONSE,
        )
        assert code == 1

    def test_prints_backend_name(self, capsys) -> None:
        _run_main_with_mocked_http(
            ["--base-url", "http://test"],
            _CAPS_CLIPSEG,
            {**_SEGMENT_RESPONSE, "backend": "clipseg"},
        )
        out = capsys.readouterr().out
        assert "clipseg" in out

    def test_prints_mask_count(self, capsys) -> None:
        _run_main_with_mocked_http(
            ["--base-url", "http://test"],
            _CAPS_MOCK,
            _SEGMENT_RESPONSE,
        )
        out = capsys.readouterr().out
        assert "Masks:" in out
        assert "1" in out

    def test_mock_backend_smoke_no_prompts(self, capsys) -> None:
        """When no prompts are given, a synthetic prompt is chosen."""
        code = _run_main_with_mocked_http(
            ["--base-url", "http://test"],  # no --text-prompt / --point / --box
            _CAPS_MOCK,
            _SEGMENT_RESPONSE,
        )
        assert code == 0

    def test_json_output_flag(self, capsys) -> None:
        _run_main_with_mocked_http(
            ["--base-url", "http://test", "--json"],
            _CAPS_MOCK,
            _SEGMENT_RESPONSE,
        )
        out = capsys.readouterr().out
        import json
        parsed = json.loads(out)
        assert parsed["backend"] == "mock"

    def test_output_dir_saves_mask(self, tmp_path) -> None:
        import json as json_mod
        seg = {
            **_SEGMENT_RESPONSE,
            "masks": [{"mask_b64": _STUB_IMAGE_B64, "score": 0.9, "area": 1, "logits_b64": None}],
        }
        _run_main_with_mocked_http(
            ["--base-url", "http://test", "--output-dir", str(tmp_path)],
            _CAPS_MOCK,
            seg,
        )
        saved = list(tmp_path.glob("mask_*.png"))
        assert len(saved) == 1


# ---------------------------------------------------------------------------
# load_image_b64
# ---------------------------------------------------------------------------

def test_load_image_b64_round_trips(tmp_path) -> None:
    img_file = tmp_path / "test.png"
    img_file.write_bytes(base64.b64decode(_STUB_IMAGE_B64))
    result = load_image_b64(str(img_file))
    assert result == _STUB_IMAGE_B64


def test_load_image_b64_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_image_b64("/nonexistent/path.png")
