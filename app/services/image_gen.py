"""
AI 图片生成服务 —— 支持 Agnes Image 2.0 Flash 和本地 ComfyUI 双后端。

与素材库（pexels/pixabay/coverr）并行共存，独立于视频下载管线。
用户只需在 config.toml 配置 backend 和对应参数，后续全部自动化：
LLM 写提示词 → 调图生 API → 存图 → 合成视频。

后端切换：config.toml [ai_image] backend = "agnes" | "comfyui"
"""

import base64
import copy
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List

import requests
from loguru import logger

from app.config import config

# ---------------------------------------------------------------------------
# Agnes Image 2.0 Flash 尺寸常量
# ---------------------------------------------------------------------------
_VALID_SIZES = ("1024x1024", "1024x768", "768x1024")
_DEFAULT_SIZE = "1024x768"

# ---------------------------------------------------------------------------
# 后端选择
# ---------------------------------------------------------------------------

_backend_cache: str | None = None


def _get_backend() -> str:
    """返回当前后端类型：'agnes' 或 'comfyui'。"""
    global _backend_cache
    backend = (config.ai_image.get("backend", "agnes") or "agnes").strip().lower()
    if backend not in ("agnes", "comfyui"):
        logger.warning(f"unknown ai_image backend '{backend}', falling back to agnes")
        backend = "agnes"
    _backend_cache = backend
    return backend


def is_enabled() -> bool:
    """检查 AI 图片生成服务是否已启用。"""
    backend = _get_backend()
    if backend == "comfyui":
        return _is_comfyui_enabled()
    return _is_agnes_enabled()


def _is_agnes_enabled() -> bool:
    """检查 Agnes 后端是否已配置。"""
    api_key = (config.ai_image.get("api_key", "") or "").strip()
    model_name = (config.ai_image.get("model_name", "") or "").strip()
    return bool(api_key and model_name)


def _is_comfyui_enabled() -> bool:
    """检查 ComfyUI 后端是否可用。"""
    server = (config.ai_image.get("comfyui_server_address", "") or "").strip()
    if not server:
        server = "http://127.0.0.1:8188"
    workflow_path = (config.ai_image.get("comfyui_workflow_path", "") or "").strip()
    if not workflow_path:
        logger.warning("ComfyUI backend enabled but comfyui_workflow_path not set")
        return False
    # 检查工作流文件是否存在
    if not os.path.isfile(workflow_path):
        logger.warning(f"ComfyUI workflow file not found: {workflow_path}")
        return False
    return True


# ---------------------------------------------------------------------------
# ComfyUI 配置读取
# ---------------------------------------------------------------------------

def _get_comfyui_server() -> str:
    return (config.ai_image.get("comfyui_server_address", "") or "").strip() or "http://127.0.0.1:8188"


def _get_comfyui_workflow_path() -> str:
    return (config.ai_image.get("comfyui_workflow_path", "") or "").strip()


def _get_comfyui_timeout() -> int:
    try:
        return int(config.ai_image.get("comfyui_timeout", 300) or 300)
    except (ValueError, TypeError):
        return 300


def _get_comfyui_batch_size() -> int:
    try:
        return int(config.ai_image.get("comfyui_batch_size", 10) or 10)
    except (ValueError, TypeError):
        return 10


def _get_comfyui_input_dir() -> str:
    """返回 ComfyUI input 目录（prompt 文件必须放这里）。"""
    # 从 workflow_path 推导 ComfyUI 根目录
    wf = _get_comfyui_workflow_path()
    # workflow_path 可能是绝对路径或相对路径
    if os.path.isabs(wf):
        # 例如 E:\...\ComfyUI\user\default\workflows\xxx.json
        # ComfyUI 根目录 = user 的上三级（4 层 dirname）
        comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(wf))))
    else:
        comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(wf))) if os.sep in wf else "."
    return os.path.join(comfyui_root, "input")


# ---------------------------------------------------------------------------
# Agnes Image 配置读取
# ---------------------------------------------------------------------------

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
    cfg_size = (config.ai_image.get("size", "") or "").strip()
    return _resolve_size(cfg_size or None)


# ---------------------------------------------------------------------------
# ComfyUI 后端实现
# ---------------------------------------------------------------------------

def _get_comfyui_client():
    """延迟导入并创建 ComfyUI 客户端实例。"""
    from app.services._comfyui_client import ComfyUIClient
    return ComfyUIClient(server_address=_get_comfyui_server())


def _generate_image_comfyui(
    prompt: str,
    output_path: str,
    **kwargs,
) -> str:
    """
    使用 ComfyUI 生成单张图片。

    原理：将单个 prompt 写入临时 .txt 文件，加载工作流、修改节点参数后提交。
    """
    try:
        client = _get_comfyui_client()

        # 检查 ComfyUI 是否可达
        if not client.ping():
            logger.error("ComfyUI server is not reachable")
            return ""

        # 加载工作流（自动识别 UI/API 格式并转换）
        workflow_path = _get_comfyui_workflow_path()
        workflow = client.load_workflow(workflow_path)

        # 将 prompt 写入 ComfyUI/input/prompts.txt（节点验证要求文件在允许列表内）
        comfyui_input = _get_comfyui_input_dir()
        os.makedirs(comfyui_input, exist_ok=True)
        prompt_file_path = os.path.join(comfyui_input, "prompts.txt")
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        # 修改工作流：使用 prompts.txt，只处理 1 个
        wf = copy.deepcopy(workflow)
        if "62" in wf:
            wf["62"]["inputs"]["file"] = "prompts.txt"
            wf["62"]["inputs"]["max_prompts"] = 1
            wf["62"]["inputs"]["start_index"] = 0
        if "63" in wf:
            wf["63"]["inputs"]["filename_prefix"] = "comfyui_single"

        timeout = _get_comfyui_timeout()
        images = client.run_workflow(wf, timeout=timeout)

        if not images:
            logger.error("ComfyUI returned no images for single prompt")
            return ""

        # 保存第一张图
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = images[0]
        with open(output_path, "wb") as f:
            f.write(img["data"])

        logger.success(f"ComfyUI image saved: {output_path}")
        return output_path

    except ImportError:
        logger.error("websocket-client is required for ComfyUI: pip install websocket-client")
        return ""
    except Exception as exc:
        logger.error(f"ComfyUI single image generation failed: {exc}")
        return ""


def _generate_images_batch_comfyui(
    task_id: str,
    prompts: List[Dict[str, Any]],
    output_dir: str,
) -> List[Dict[str, Any]]:
    """
    使用 ComfyUI 批量生成图片。

    原理：将所有 prompts 写入一个临时 .txt 文件，加载工作流后一次性提交。
    ComfyUI 的 PromptLoopFromFile 节点会逐行读取并生成多张图。
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    # 检查哪些图片需要生成（跳过已存在的）
    pending_items: List[Dict[str, Any]] = []
    for item in prompts:
        idx = item["index"]
        image_path = os.path.join(output_dir, f"scene-{idx:02d}.png")
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            logger.info(f"[{task_id}] image already exists, skipping: {image_path}")
            item["image_path"] = image_path
            results.append(item)
        else:
            item["image_path"] = ""
            pending_items.append(item)

    if not pending_items:
        logger.info(f"[{task_id}] all images already exist, nothing to generate")
        return results

    try:
        client = _get_comfyui_client()

        # 检查 ComfyUI 是否可达
        if not client.ping():
            logger.error(f"[{task_id}] ComfyUI server is not reachable")
            # 所有待生成项标记为失败
            for item in pending_items:
                results.append(item)
            return results

        # 加载工作流（自动识别 UI/API 格式并转换）
        workflow_path = _get_comfyui_workflow_path()
        workflow = client.load_workflow(workflow_path)

        # 将所有 prompts 写入 ComfyUI/input/prompts.txt
        comfyui_input = _get_comfyui_input_dir()
        os.makedirs(comfyui_input, exist_ok=True)
        prompt_file_path = os.path.join(comfyui_input, "prompts.txt")
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            for item in pending_items:
                f.write(item["prompt"] + "\n")

        # 修改工作流参数
        wf = copy.deepcopy(workflow)
        if "62" in wf:
            wf["62"]["inputs"]["file"] = "prompts.txt"
            wf["62"]["inputs"]["max_prompts"] = len(pending_items)
            wf["62"]["inputs"]["start_index"] = 0
        if "63" in wf:
            wf["63"]["inputs"]["filename_prefix"] = f"mpt_{task_id}"

        timeout = _get_comfyui_timeout()
        logger.info(
            f"[{task_id}] ComfyUI batch: submitting {len(pending_items)} "
            f"prompts, timeout={timeout}s"
        )
        images = client.run_workflow(wf, timeout=timeout)

        if not images:
            logger.error(f"[{task_id}] ComfyUI returned no images")
            for item in pending_items:
                results.append(item)
            return results

        logger.info(f"[{task_id}] ComfyUI generated {len(images)} images")

        # 将 ComfyUI 输出的图片按顺序映射到预期的文件路径
        for i, item in enumerate(pending_items):
            if i < len(images):
                idx = item["index"]
                image_path = os.path.join(output_dir, f"scene-{idx:02d}.png")
                img = images[i]
                with open(image_path, "wb") as f:
                    f.write(img["data"])
                item["image_path"] = image_path
                logger.info(f"[{task_id}] scene-{idx:02d}.png saved")
            else:
                logger.warning(
                    f"[{task_id}] not enough images from ComfyUI for "
                    f"segment {item['index'] + 1}"
                )
            results.append(item)

    except ImportError:
        logger.error("websocket-client is required for ComfyUI: pip install websocket-client")
        for item in pending_items:
            results.append(item)
    except Exception as exc:
        logger.error(f"[{task_id}] ComfyUI batch generation failed: {exc}")
        for item in pending_items:
            results.append(item)

    return results


# ---------------------------------------------------------------------------
# Agnes Image 后端实现（原始逻辑）
# ---------------------------------------------------------------------------

def _generate_image_agnes(
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
            img_resp = requests.get(image_data, timeout=60)
            img_resp.raise_for_status()
            png_bytes = img_resp.content
        else:
            png_bytes = base64.b64decode(image_data)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        logger.success(f"image saved: {output_path}")
        return output_path
    except Exception as exc:
        logger.error(f"failed to save image: {exc}")
        return ""


def _generate_images_batch_agnes(
    task_id: str,
    prompts: List[Dict[str, Any]],
    output_dir: str,
) -> List[Dict[str, Any]]:
    """使用 Agnes API 逐条生成图片。"""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for item in prompts:
        idx = item["index"]
        image_path = os.path.join(output_dir, f"scene-{idx:02d}.png")
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            logger.info(f"image already exists, skipping: {image_path}")
            item["image_path"] = image_path
            results.append(item)
            continue
        success_path = _generate_image_agnes(item["prompt"], image_path)
        item["image_path"] = success_path or ""
        results.append(item)
        if not success_path:
            logger.warning(f"failed to generate image for segment {idx + 1}")

    return results


# ---------------------------------------------------------------------------
# 统一对外接口（路由到具体后端）
# ---------------------------------------------------------------------------

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
    生成单张图片并保存到 output_path。

    根据 config.toml [ai_image] backend 自动路由到 Agnes 或 ComfyUI。

    Returns:
        实际保存的文件路径，失败返回空字符串。
    """
    backend = _get_backend()
    if backend == "comfyui":
        return _generate_image_comfyui(prompt, output_path)
    return _generate_image_agnes(
        prompt,
        output_path,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        size=size,
    )


def generate_images_batch(
    task_id: str,
    prompts: List[Dict[str, Any]],
    output_dir: str,
) -> List[Dict[str, Any]]:
    """
    批量生成图片，为每段素材生成对应图片。

    根据 config.toml [ai_image] backend 自动路由到 Agnes 或 ComfyUI。
    ComfyUI 后端会将所有 prompts 写入临时文件后一次性提交工作流。

    Args:
        task_id: 任务 ID（仅用于日志）
        prompts: generate_image_prompts() 返回的列表，每项含 index/segment/prompt
        output_dir: 图片输出目录

    Returns:
        更新后的 prompts 列表，每项新增 "image_path" 字段；生成失败的项 image_path 为空。
    """
    backend = _get_backend()
    if backend == "comfyui":
        return _generate_images_batch_comfyui(task_id, prompts, output_dir)
    return _generate_images_batch_agnes(task_id, prompts, output_dir)
