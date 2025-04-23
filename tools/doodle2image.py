from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from typing import Generator, Any, Dict
import requests
import time

class Doodle2Image(Tool):
    # 轮询间隔（秒）
    POLL_INTERVAL = 2
    # 最大轮询次数
    MAX_POLL_RETRIES = 900
    # API 超时时间（秒）
    API_TIMEOUT = 30

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            # 验证参数
            self._validate_parameters(tool_parameters)
            api_key = self.runtime.credentials.get("tonyi_api_key")
            
            # 提交生成任务
            task_id = self._submit_image_task(api_key, tool_parameters)
            # 轮询任务结果
            yield from self._poll_task_status(api_key, task_id)
            
        except Exception as e:
            yield self.create_text_message(str(e))

    def _submit_image_task(self, api_key: str, params: Dict) -> str:
        """提交图片生成任务并返回 task_id"""
        try:
            response = requests.post(
                url="https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis",
                headers={
                    "X-DashScope-Async": "enable",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": params["model"],
                    "input": {
                        "sketch_image_url": params["sketch_image_url"],
                        "prompt": params["prompt"]
                    },
                    "parameters": {
                        "style": params["style"],
                        "size": params["size"],
                        "n": params["count"],
                        "sketch_weight": params["sketch_weight"]
                    }
                },
                timeout=self.API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            if not (result.get("output") and result["output"].get("task_id")):
                raise ValueError("Invalid API response: missing task_id")
                
            return result["output"]["task_id"]
        
        except requests.exceptions.RequestException as e:
            try:
                # 尝试从响应中获取错误信息
                error_response = e.response.json()
                code = error_response.get("code", "Unknown error code")
                message = error_response.get("message", "No error message provided")
                
                # 返回错误提示
                raise RuntimeError(f"Sorry, an error occurred: {message} (Error Code: {code}). Please try again later or contact support.")
            
            except ValueError:
                # 如果无法解析错误响应，提供通用错误信息
                raise RuntimeError("Sorry, an unknown error occurred. Please try again later or contact support.")

    def _poll_task_status(self, api_key: str, task_id: str) -> Generator[ToolInvokeMessage, None, None]:
        """轮询任务状态并返回结果"""
        for _ in range(self.MAX_POLL_RETRIES):
            task_data = self._fetch_task_result(api_key, task_id)
            status = task_data.get("output", {}).get("task_status")
            message = task_data.get("output", {}).get("message")

            if status == "SUCCEEDED":
                yield from self._handle_success_result(task_data, task_id)
                return
            elif status in ["FAILED", "UNKNOWN"]:
                raise RuntimeError(f"Task {task_id} failed with status: {status}. Message: {message}")
            
            time.sleep(self.POLL_INTERVAL)
        
        raise TimeoutError("Task processing timeout")

    def _fetch_task_result(self, api_key: str, task_id: str) -> Dict:
        """获取任务结果"""
        response = requests.get(
            url=f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def _handle_success_result(self, task_data: Dict, task_id: str) -> Generator[ToolInvokeMessage, None, None]:
        """处理成功结果"""
        results = task_data.get("output", {}).get("results", [])
        yield self.create_json_message({
            "task_id": task_id,
            "image_count": len(results),
            "results": results
        })
        
        for item in results:
            if url := item.get("url"):
                yield self.create_image_message(url)

    def _validate_parameters(self, params: Dict) -> None:
        """验证必要参数"""
        if not params.get("model"):
            raise ValueError("Missing required parameter: Model")
        if not params.get("prompt"):
            raise ValueError("Missing required parameter: Prompt")
        if not params.get("sketch_image_url"):
            raise ValueError("Missing required parameter: Sketch Image URL")