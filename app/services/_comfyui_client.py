"""
ComfyUI API 客户端 —— 封装与本地 ComfyUI 实例的交互。

支持 UI 格式和 API 格式工作流自动识别与转换。
支持通过 WebSocket 监听工作流执行状态。
依赖：websocket-client（pip install websocket-client）
"""

import copy
import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import websocket  # type: ignore
except ImportError:
    websocket = None  # type: ignore


# ======================================================================
# UI 格式 → API 格式 转换
# ======================================================================

# 已知节点类型的 widgets_values → inputs 名称映射
_WIDGET_MAP: Dict[str, List[str]] = {
    "PromptLoopFromFile": ["file", "start_index", "max_prompts"],
    "PromptLoopSaveImage": ["filename_prefix", "negative_prompt"],
    "DF_String_Concatenate": ["text_a", "text_b"],
}


def _is_api_format(workflow: dict) -> bool:
    """判断工作流是否为 API 格式（顶层 key 是数字节点 ID，值含 class_type）。"""
    if "nodes" in workflow:
        return False
    for key, val in workflow.items():
        if isinstance(val, dict) and "class_type" in val:
            return True
    return False


def _ui_to_api(workflow: dict) -> dict:
    """
    将 ComfyUI UI 格式工作流转换为 API 格式。

    UI 格式特征：
      - 顶层有 "nodes" 数组，每个节点有 id / type / widgets_values / inputs / outputs
      - 顶层有 "links" 数组，描述节点间连接

    API 格式特征：
      - 顶层 key 是字符串化的节点 ID，值为 {"class_type": ..., "inputs": {...}}
      - 连接用 ["node_id", output_slot] 表示
    """
    nodes = {n["id"]: n for n in workflow.get("nodes", [])}
    links = workflow.get("links", [])

    # 建立连接映射：(to_node, to_slot) → (from_node, from_slot)
    connection_map: Dict[tuple, tuple] = {}
    for link in links:
        # link 格式: [link_id, from_node, from_slot, to_node, to_slot, type_str]
        if len(link) >= 5:
            _, from_node, from_slot, to_node, to_slot = link[:5]
            connection_map[(to_node, to_slot)] = (from_node, from_slot)

    api_workflow: Dict[str, Any] = {}

    for node_id, node in nodes.items():
        node_type = node.get("type", "")
        widgets = node.get("widgets_values", [])

        inputs: Dict[str, Any] = {}

        # 1) 将 widgets_values 映射为命名 inputs
        widget_names = _WIDGET_MAP.get(node_type, [])
        for i, name in enumerate(widget_names):
            if i < len(widgets):
                inputs[name] = widgets[i]

        # 2) 将连接信息映射为 inputs（覆盖同名 widget，因为连接优先）
        #    需要从节点的 inputs/slot 信息中找到 slot 索引对应的 input 名称
        node_inputs = node.get("inputs", [])
        for slot_idx, inp_def in enumerate(node_inputs):
            inp_name = inp_def.get("name", f"input_{slot_idx}")
            if (node_id, slot_idx) in connection_map:
                from_node, from_slot = connection_map[(node_id, slot_idx)]
                inputs[inp_name] = [str(from_node), from_slot]

        # 3) 子图节点需要特殊处理：读取子图中的子工作流
        if node_type == "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1":
            # 这是一个子图节点，需要从 extra 或其他地方获取子工作流信息
            # 将 widgets_values 作为 inputs
            if not inputs and widgets:
                inputs = {"value": widgets}

        api_workflow[str(node_id)] = {
            "class_type": node_type,
            "inputs": inputs,
        }

    return api_workflow


def _load_and_normalize_workflow(workflow_path: str) -> dict:
    """
    加载工作流文件，自动检测格式并统一转为 API 格式。
    """
    with open(workflow_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if _is_api_format(data):
        logger.debug(f"workflow is already API format: {workflow_path}")
        return data

    logger.info(f"workflow is UI format, converting to API: {workflow_path}")
    api = _ui_to_api(data)
    logger.debug(f"converted {len(api)} nodes to API format")
    return api


# ======================================================================
# ComfyUI 客户端
# ======================================================================

class ComfyUIClient:
    """与本地 ComfyUI 实例交互的客户端。"""

    def __init__(self, server_address: str = "http://127.0.0.1:8188"):
        addr = server_address.rstrip("/")
        if not addr.startswith("http://") and not addr.startswith("https://"):
            addr = f"http://{addr}"
        self.server_address = addr
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # HTTP 工具
    # ------------------------------------------------------------------

    def _http_post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.server_address}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get_json(self, path: str) -> dict:
        url = f"{self.server_address}{path}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get_bytes(self, path: str, params: Optional[dict] = None) -> bytes:
        url = f"{self.server_address}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()

    # ------------------------------------------------------------------
    # 基础 API
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """检查 ComfyUI 是否可达。"""
        try:
            self._http_get_json("/system_stats")
            return True
        except Exception as exc:
            logger.warning(f"ComfyUI ping failed: {exc}")
            return False

    def queue_prompt(self, prompt: dict) -> str:
        """
        提交工作流到 ComfyUI 队列。
        prompt 必须是 API 格式。
        """
        payload = {"prompt": prompt, "client_id": self.client_id}
        result = self._http_post_json("/prompt", payload)
        prompt_id = result.get("prompt_id", "")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI queue_prompt returned no prompt_id: {result}")
        logger.info(f"ComfyUI prompt queued: {prompt_id}")
        return prompt_id

    def get_history(self, prompt_id: str) -> dict:
        return self._http_get_json(f"/history/{prompt_id}")

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        return self._http_get_bytes("/view", params)

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> bool:
        """
        通过 WebSocket 等待工作流执行完成。

        Args:
            prompt_id: 由 queue_prompt 返回的任务 ID。
            timeout: 最大等待秒数（整个工作流，非单节点）。

        Returns:
            True = 正常完成，False = 超时。
        """
        if websocket is None:
            raise ImportError(
                "websocket-client is required. Install: pip install websocket-client"
            )

        ws_url = self.server_address.replace("http://", "ws://").replace("https://", "wss://")
        ws = websocket.WebSocket()
        ws.connect(f"{ws_url}/ws?clientId={self.client_id}")
        try:
            start = time.time()
            while True:
                elapsed = time.time() - start
                if elapsed > timeout:
                    logger.error(f"ComfyUI timed out after {timeout}s (prompt_id={prompt_id})")
                    return False
                ws.settimeout(max(1, timeout - elapsed))
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "executing":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id and data.get("node") is None:
                            logger.info(f"ComfyUI workflow completed: {prompt_id}")
                            return True
        finally:
            ws.close()

    # ------------------------------------------------------------------
    # 高层封装
    # ------------------------------------------------------------------

    def load_workflow(self, workflow_path: str) -> dict:
        """
        加载工作流文件，自动检测 UI/API 格式并统一转为 API 格式。
        """
        return _load_and_normalize_workflow(workflow_path)

    def run_workflow(self, workflow: dict, timeout: int = 300) -> List[Dict[str, Any]]:
        """
        提交工作流并等待完成，返回所有输出图片。

        workflow 可以是 UI 格式或 API 格式，会自动转换。
        """
        # 确保是 API 格式
        if not _is_api_format(workflow):
            workflow = _ui_to_api(workflow)

        prompt_id = self.queue_prompt(workflow)
        ok = self.wait_for_completion(prompt_id, timeout=timeout)
        if not ok:
            raise TimeoutError(f"ComfyUI workflow timed out: {prompt_id}")

        history = self.get_history(prompt_id)
        images: List[Dict[str, Any]] = []
        for node_id, node_output in history.get(prompt_id, {}).get("outputs", {}).items():
            for img in node_output.get("images", []):
                img_data = self.get_image(img["filename"], img["subfolder"], img["type"])
                images.append({
                    "filename": img["filename"],
                    "subfolder": img["subfolder"],
                    "type": img["type"],
                    "data": img_data,
                })
        return images

    def generate_from_workflow_file(
        self,
        workflow_path: str,
        *,
        output_dir: str = "",
        timeout: int = 300,
    ) -> List[str]:
        """加载工作流文件并执行，保存输出图片。"""
        workflow = self.load_workflow(workflow_path)
        if not output_dir:
            output_dir = str(Path(workflow_path).parent / "output")
        os.makedirs(output_dir, exist_ok=True)

        images = self.run_workflow(workflow, timeout=timeout)

        saved: List[str] = []
        for img in images:
            filepath = os.path.join(output_dir, img["filename"])
            with open(filepath, "wb") as f:
                f.write(img["data"])
            saved.append(filepath)
            logger.info(f"ComfyUI image saved: {filepath}")
        return saved
