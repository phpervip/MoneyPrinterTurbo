"""
AI 图片生成服务 —— 通过 Agnes Image 2.0 Flash（OpenAI 兼容）批量生成视频素材图片。

与素材库（pexels/pixabay/coverr）并行共存，独立于视频下载管线。
用户只需在 config.toml 配置 API Key 和模型，后续全部自动化：
LLM 写提示词 → 调图生 API → 存图 → 合成视频。
"""

import base64
import os
from typing import Any, Dict, List

import requests
from loguru import logger

from app.config import config

# Agnes Image 2.0 Flash 官方支持的尺寸，直接透传给 API。
# 不传无效尺寸（如 1080x1920），否则 API 会返回错误。
_VALID_SIZES = ("1024x1024", "1024x768", "768x1024")
# 竖屏视频（9:16）推荐用 768x1024，横向用 1024x768，方形用 1024x1024。
_DEFAULT_SIZE = "1024x768"


def is_enabled() -> bool:
    """检查 AI 图片生成服务是否已启用。"""
    api_key = (config.ai_image.get("api_key", "") or "").strip()
    model_name = (config.ai_image.get("model_name", "") or "").strip()
    return bool(api_key and model_name)


def get_api_key() -> str:
    return (config.ai_image.get("api_key", "") or "").strip()


def get_base_url() -> str:
    return (
        (config.ai_image.get("base_url", "") or "").strip()
        or "https://apihub.agnes-ai.com/v1"
    )


def get_model_name() -> str:
    return (
        (config.ai_image.get("model_name", "") or "").strip()
        or "agnes-image-2.0-flash"
    )


def _resolve_size(size: str | None) -> str:
    """校验并返回合法尺寸，非法则回退到默认。"""
    if size and size in _VALID_SIZES:
        return size
    if size:
        logger.warning(f"unsupported size '{size}', falling back to {_DEFAULT_SIZE}")
    # 优先读 config 里的 size 字段；没有则用竖屏默认
    cfg_size = (config.ai_image.get("size", "") or "").strip()
    return _resolve_size(cfg_size or None)


def generate_image(
    prompt: str,
    output_path: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    size: str | None = None,
) -> str:
    """
    调用 Agnes Image 2.0 Flash API 生成图片并保存到 output_path。

    使用 requests 直连而非 OpenAI SDK，原因是 Agnes API 要求
    response_format 必须嵌套在 extra_body 内，OpenAI SDK 无法透传此结构。

    Returns:
        实际保存的文件路径，失败返回空字符串。
    """
    api_key = api_key or get_api_key()
    base_url = base_url or get_base_url()
    model_name = model_name or get_model_name()
    size = _resolve_size(size)

    if not api_key:
        logger.error("AI image generation: api_key not configured")
        return ""
    if not model_name:
        logger.error("AI image generation: model_name not configured")
        return ""

    url = f"{base_url.rstrip('/')}/images/generations"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "size": size,
        "extra_body": {
            "response_format": "url",
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        logger.info(
            f"generating image: model={model_name}, size={size}, "
            f"prompt={prompt[:80]}..."
        )
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        status = getattr(exc, "response", None)
        body = status.text if status is not None else str(exc)
        logger.error(f"AI image generation HTTP error {exc}: {body[:300]}")
        return ""
    except Exception as exc:
        logger.error(f"AI image generation request failed: {exc}")
        return ""

    data = resp.json() if resp.text else {}
    items = data.get("data", [])
    if not items:
        logger.error("AI image generation: empty data in response")
        logger.debug(f"full response: {data}")
        return ""

    image_data = items[0].get("b64_json") or items[0].get("url")
    if not image_data:
        logger.error("AI image generation: no b64_json or url in response")
        logger.debug(f"response item: {items[0]}")
        return ""

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if image_data.startswith("http"):
            # URL 输出：下载图片
            img_resp = requests.get(image_data, timeout=60)
            img_resp.raise_for_status()
            png_bytes = img_resp.content
        else:
            # Base64 输出
            png_bytes = base64.b64decode(image_data)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        logger.success(f"image saved: {output_path}")
        return output_path
    except Exception as exc:
        logger.error(f"failed to save image: {exc}")
        return ""


def generate_images_batch(
    task_id: str,
    prompts: List[Dict[str, Any]],
    output_dir: str,
) -> List[Dict[str, Any]]:
    """
    批量生成图片，为每段素材生成对应图片。

    Args:
        task_id: 任务 ID（仅用于日志）
        prompts: generate_image_prompts() 返回的列表，每项含 index/segment/prompt
        output_dir: 图片输出目录

    Returns:
        更新后的 prompts 列表，每项新增 "image_path" 字段；生成失败的项 image_path 为空。
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for item in prompts:
        idx = item["index"]
        image_path = os.path.join(output_dir, f"scene-{idx:02d}.png")
        # 跳过已存在的图片
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            logger.info(f"image already exists, skipping: {image_path}")
            item["image_path"] = image_path
            results.append(item)
            continue
        success_path = generate_image(item["prompt"], image_path)
        item["image_path"] = success_path or ""
        results.append(item)
        if not success_path:
            logger.warning(f"failed to generate image for segment {idx + 1}")

    return results
