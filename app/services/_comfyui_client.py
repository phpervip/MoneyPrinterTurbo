"""
ComfyUI API 客户端 —— 封装与本地 ComfyUI 实例的交互。

支持通过 WebSocket 监听工作流执行状态，提交工作流、等待完成、下载生成的图片。
依赖：websocket-client（pip install websocket-client）
"""

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


class ComfyUIClient:
    """与本地 ComfyUI 实例交互的客户端。"""

    def __init__(self, server_address: str = "http://127.0.0.1:8188"):
        """
        初始化客户端。

        Args:
            server_address: ComfyUI 服务地址，支持 http:// 或不带协议的 host:port。
        """
        # 归一化地址：去掉末尾斜杠，确保有协议前缀
        addr = server_address.rstrip("/")
        if not addr.startswith("http://") and not addr.startswith("https://"):
            addr = f"http://{addr}"
        self.server_address = addr
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _http_post_json(self, path: str, payload: dict) -> dict:
        """POST JSON 到 ComfyUI 并返回解析后的响应。"""
        url = f"{self.server_address}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get_json(self, path: str) -> dict:
        """GET 请求 ComfyUI 并返回解析后的 JSON。"""
        url = f"{self.server_address}{path}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get_bytes(self, path: str, params: Optional[dict] = None) -> bytes:
        """GET 请求 ComfyUI 并返回原始字节。"""
        url = f"{self.server_address}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()

    # ------------------------------------------------------------------
    # 公开 API
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

        Args:
            prompt: ComfyUI API 格式的工作流字典。

        Returns:
            prompt_id，后续用于查询状态。
        """
        payload = {"prompt": prompt, "client_id": self.client_id}
        result = self._http_post_json("/prompt", payload)
        prompt_id = result.get("prompt_id", "")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI queue_prompt returned no prompt_id: {result}")
        logger.info(f"ComfyUI prompt queued: {prompt_id}")
        return prompt_id

    def get_history(self, prompt_id: str) -> dict:
        """获取指定 prompt_id 的执行历史。"""
        return self._http_get_json(f"/history/{prompt_id}")

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """从 ComfyUI 下载一张已生成的图片。"""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        return self._http_get_bytes("/view", params)

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> bool:
        """
        通过 WebSocket 等待工作流执行完成。

        Args:
            prompt_id: 由 queue_prompt 返回的任务 ID。
            timeout: 最大等待秒数。

        Returns:
            True 表示正常完成，False 表示超时。
        """
        if websocket is None:
            raise ImportError(
                "websocket-client is required for ComfyUI integration. "
                "Install it with: pip install websocket-client"
            )

        ws_url = self.server_address.replace("http://", "ws://").replace("https:// "wss://")
        ws = websocket.WebSocket()
        ws.connect(f"{ws_url}/ws?clientId={self.client_id}")
        try:
            start = time.time()
            while True:
                if time.time() - start > timeout:
                    logger.error(f"ComfyUI wait_for_completion timed out after {timeout}s")
                    return False
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

    def run_workflow(
        self,
        workflow: dict,
        timeout: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        提交工作流并等待完成，返回所有输出图片。

        Args:
            workflow: ComfyUI API 格式的工作流字典。
            timeout: 最大等待秒数。

        Returns:
            图片列表，每项包含 filename / subfolder / type / data(字节)。
        """
        prompt_id = self.queue_prompt(workflow)
        ok = self.wait_for_completion(prompt_id, timeout=timeout)
        if not ok:
            raise TimeoutError(f"ComfyUI workflow timed out: {prompt_id}")

        history = self.get_history(prompt_id)
        images: List[Dict[str, Any]] = []
        for node_id, node_output in history.get(prompt_id, {}).get("outputs", {}).items():
            for img in node_output.get("images", []):
                img_data = self.get_image(img["filename"], img["subfolder"], img["type"])
                images.append(
                    {
                        "filename": img["filename"],
                        "subfolder": img["subfolder"],
                        "type": img["type"],
                        "data": img_data,
                    }
                )
        return images

    def generate_from_workflow_file(
        self,
        workflow_path: str,
        *,
        output_dir: str = "",
        timeout: int = 300,
    ) -> List[str]:
        """
        加载工作流 JSON 文件并执行，将输出图片保存到指定目录。

        Args:
            workflow_path: 工作流 JSON 文件的绝对或相对路径。
            output_dir: 图片保存目录，为空则保存到工作流同目录下的 output/。
            timeout: 最大等待秒数。

        Returns:
            保存的文件路径列表。
        """
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

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

    def generate_from_prompt_file(
        self,
        workflow: dict,
        prompt_file: str,
        *,
        max_prompts: int = 10,
        filename_prefix: str = "scene",
        timeout: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        修改工作流中的 PromptLoopFromFile 和 PromptLoopSaveImage 节点后执行。

        这是一个高层封装，适用于 image_z_image_turbo_int8_2_batch.json 这类
        使用 PromptLoopFromFile → PromptLoopSaveImage 的工作流。

        Args:
            workflow: 工作流字典（会被原地修改）。
            prompt_file: 提示词文件名（ComfyUI 可访问的路径）。
            max_prompts: 每批处理的提示词数量。
            filename_prefix: 生成图片的文件名前缀。
            timeout: 最大等待秒数。

        Returns:
            生成的图片列表。
        """
        import copy
        wf = copy.deepcopy(workflow)

        # 修改 PromptLoopFromFile 节点（节点 ID 62）
        if "62" in wf:
            wf["62"]["inputs"]["file"] = prompt_file
            wf["62"]["inputs"]["max_prompts"] = max_prompts

        # 修改 PromptLoopSaveImage 节点（节点 ID 63）
        if "63" in wf:
            wf["63"]["inputs"]["filename_prefix"] = filename_prefix

        return self.run_workflow(wf, timeout=timeout)
