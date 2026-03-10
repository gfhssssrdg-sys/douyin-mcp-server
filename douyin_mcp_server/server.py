#!/usr/bin/env python3
"""
抖音无水印视频下载并提取文本的 MCP 服务器

支持：
1. 解析抖音分享链接获取无水印视频链接
2. 直接返回无水印下载链接
3. 使用 DashScope 或火山引擎 / 火山方舟 ASR 提取文案
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import dashscope
import requests
from http import HTTPStatus
from urllib import request

from mcp.server.fastmcp import Context, FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP(
    "Douyin MCP Server",
    dependencies=["requests", "dashscope"],
)

# 请求头，模拟移动端访问
HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1"
}

DEFAULT_PROVIDER = "dashscope"
DEFAULT_DASHSCOPE_MODEL = "paraformer-v2"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-asr-1-0"


class DouyinProcessor:
    """抖音视频处理器"""

    def __init__(
        self,
        api_key: str = "",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.provider = (provider or DEFAULT_PROVIDER).lower()
        self.model = model or self._default_model(self.provider)
        self.api_base_url = api_base_url or self._default_base_url(self.provider)
        if self.provider == "dashscope" and api_key:
            dashscope.api_key = api_key

    @staticmethod
    def _default_model(provider: str) -> str:
        return DEFAULT_ARK_MODEL if provider == "volcengine" else DEFAULT_DASHSCOPE_MODEL

    @staticmethod
    def _default_base_url(provider: str) -> Optional[str]:
        return DEFAULT_ARK_BASE_URL if provider == "volcengine" else None

    def parse_share_url(self, share_text: str) -> dict:
        """从分享文本中提取无水印视频链接"""
        urls = re.findall(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", share_text)
        if not urls:
            raise ValueError("未找到有效的分享链接")

        share_url = urls[0]
        share_response = requests.get(share_url, headers=HEADERS)
        video_id = share_response.url.split("?")[0].strip("/").split("/")[-1]
        share_url = f"https://www.iesdouyin.com/share/video/{video_id}"

        response = requests.get(share_url, headers=HEADERS)
        response.raise_for_status()

        pattern = re.compile(pattern=r"window\._ROUTER_DATA\s*=\s*(.*?)</script>", flags=re.DOTALL)
        find_res = pattern.search(response.text)

        if not find_res or not find_res.group(1):
            raise ValueError("从HTML中解析视频信息失败")

        json_data = json.loads(find_res.group(1).strip())
        video_page_key = "video_(id)/page"
        note_page_key = "note_(id)/page"

        if video_page_key in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][video_page_key]["videoInfoRes"]
        elif note_page_key in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][note_page_key]["videoInfoRes"]
        else:
            raise Exception("无法从JSON中解析视频或图集信息")

        data = original_video_info["item_list"][0]
        video_url = data["video"]["play_addr"]["url_list"][0].replace("playwm", "play")
        desc = data.get("desc", "").strip() or f"douyin_{video_id}"
        desc = re.sub(r'[\\/:*?"<>|]', "_", desc)

        return {"url": video_url, "title": desc, "video_id": video_id}

    def extract_text_from_video_url(self, video_url: str) -> str:
        if self.provider == "dashscope":
            return self._extract_text_dashscope(video_url)
        if self.provider == "volcengine":
            return self._extract_text_volcengine(video_url)
        raise ValueError(f"不支持的 provider: {self.provider}，可选值: dashscope / volcengine")

    def _extract_text_dashscope(self, video_url: str) -> str:
        try:
            task_response = dashscope.audio.asr.Transcription.async_call(
                model=self.model,
                file_urls=[video_url],
                language_hints=["zh", "en"],
            )
            transcription_response = dashscope.audio.asr.Transcription.wait(task=task_response.output.task_id)

            if transcription_response.status_code != HTTPStatus.OK:
                raise Exception(transcription_response.output.message)

            for transcription in transcription_response.output["results"]:
                url = transcription["transcription_url"]
                result = json.loads(request.urlopen(url).read().decode("utf8"))
                if result.get("transcripts"):
                    return result["transcripts"][0].get("text", "") or "未识别到文本内容"
            return "未识别到文本内容"
        except Exception as e:
            raise Exception(f"DashScope 转录失败: {str(e)}")

    def _extract_text_volcengine(self, video_url: str) -> str:
        endpoint = f"{self.api_base_url.rstrip('/')}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "file_url": video_url}

        try:
            response = requests.post(endpoint, headers=headers, data=data, timeout=300)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            raise Exception(f"火山引擎 / 方舟转录失败: {str(e)}")

        text = result.get("text") or result.get("result") or result.get("transcript")
        if text:
            return text
        return json.dumps(result, ensure_ascii=False)


def resolve_runtime_config(provider: Optional[str], model: Optional[str], api_base_url: Optional[str]) -> dict:
    resolved_provider = (provider or os.getenv("ASR_PROVIDER") or os.getenv("TRANSCRIPTION_PROVIDER") or DEFAULT_PROVIDER).lower()

    if resolved_provider == "volcengine":
        resolved_api_key = (
            os.getenv("ARK_API_KEY")
            or os.getenv("VOLCENGINE_API_KEY")
            or os.getenv("API_KEY")
        )
        resolved_model = model or os.getenv("ARK_ASR_MODEL") or os.getenv("ARK_MODEL") or os.getenv("VOLCENGINE_ASR_MODEL") or DEFAULT_ARK_MODEL
        resolved_base_url = api_base_url or os.getenv("ARK_BASE_URL") or os.getenv("VOLCENGINE_BASE_URL") or DEFAULT_ARK_BASE_URL
    else:
        resolved_provider = "dashscope"
        resolved_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
        resolved_model = model or os.getenv("DASHSCOPE_ASR_MODEL") or os.getenv("DASHSCOPE_MODEL") or DEFAULT_DASHSCOPE_MODEL
        resolved_base_url = api_base_url

    return {
        "provider": resolved_provider,
        "api_key": resolved_api_key,
        "model": resolved_model,
        "api_base_url": resolved_base_url,
    }


@mcp.tool()
def get_douyin_download_link(share_link: str) -> str:
    """获取抖音视频的无水印下载链接。"""
    try:
        processor = DouyinProcessor()
        video_info = processor.parse_share_url(share_link)
        return json.dumps(
            {
                "status": "success",
                "video_id": video_info["video_id"],
                "title": video_info["title"],
                "download_url": video_info["url"],
                "description": f"视频标题: {video_info['title']}",
                "usage_tip": "可以直接使用此链接下载无水印视频",
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": f"获取下载链接失败: {str(e)}"}, ensure_ascii=False, indent=2)


@mcp.tool()
async def extract_douyin_text(
    share_link: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_base_url: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    从抖音分享链接提取视频中的文本内容。

    参数:
    - share_link: 抖音分享链接或包含链接的文本
    - model: 语音识别模型（可选）
    - provider: ASR 提供方，可选 dashscope / volcengine
    - api_base_url: 自定义 API Base URL（主要用于火山引擎 / 方舟兼容网关）
    """
    try:
        config = resolve_runtime_config(provider, model, api_base_url)
        if not config["api_key"]:
            if config["provider"] == "volcengine":
                raise ValueError("未设置火山引擎 / 方舟 API Key，请配置 ARK_API_KEY 或 VOLCENGINE_API_KEY")
            raise ValueError("未设置 DashScope API Key，请配置 DASHSCOPE_API_KEY 或兼容的 API_KEY")

        processor = DouyinProcessor(**config)

        if ctx:
            ctx.info("正在解析抖音分享链接...")
        video_info = processor.parse_share_url(share_link)

        if ctx:
            ctx.info(f"正在使用 {config['provider']} 提取文本...")
        text_content = processor.extract_text_from_video_url(video_info["url"])

        if ctx:
            ctx.info("文本提取完成!")
        return text_content
    except Exception as e:
        if ctx:
            ctx.error(f"处理过程中出现错误: {str(e)}")
        raise Exception(f"提取抖音视频文本失败: {str(e)}")


@mcp.tool()
def parse_douyin_video_info(share_link: str) -> str:
    """解析抖音分享链接，获取视频基本信息。"""
    try:
        processor = DouyinProcessor()
        video_info = processor.parse_share_url(share_link)
        return json.dumps(
            {
                "video_id": video_info["video_id"],
                "title": video_info["title"],
                "download_url": video_info["url"],
                "status": "success",
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2)


@mcp.resource("douyin://video/{video_id}")
def get_video_info(video_id: str) -> str:
    """获取指定视频ID的详细信息。"""
    share_url = f"https://www.iesdouyin.com/share/video/{video_id}"
    try:
        processor = DouyinProcessor()
        video_info = processor.parse_share_url(share_url)
        return json.dumps(video_info, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取视频信息失败: {str(e)}"


@mcp.prompt()
def douyin_text_extraction_guide() -> str:
    """抖音视频文本提取使用指南"""
    return f"""
# 抖音视频文本提取使用指南

## 支持的 ASR Provider
- `dashscope`（默认）
- `volcengine`（火山引擎 / 火山方舟，OpenAI-compatible Audio Transcriptions）

## 环境变量配置
### DashScope
- `ASR_PROVIDER=dashscope`（可省略）
- `DASHSCOPE_API_KEY` 或 `API_KEY`
- `DASHSCOPE_ASR_MODEL`（默认 `{DEFAULT_DASHSCOPE_MODEL}`）

### 火山引擎 / 火山方舟
- `ASR_PROVIDER=volcengine`
- `ARK_API_KEY` 或 `VOLCENGINE_API_KEY`
- `ARK_BASE_URL`（默认 `{DEFAULT_ARK_BASE_URL}`）
- `ARK_ASR_MODEL` / `ARK_MODEL`（默认 `{DEFAULT_ARK_MODEL}`）

## Claude Desktop 配置示例
### DashScope
```json
{{
  "mcpServers": {{
    "douyin-mcp": {{
      "command": "uvx",
      "args": ["douyin-mcp-server"],
      "env": {{
        "ASR_PROVIDER": "dashscope",
        "DASHSCOPE_API_KEY": "your-dashscope-key"
      }}
    }}
  }}
}}
```

### 火山引擎 / 火山方舟
```json
{{
  "mcpServers": {{
    "douyin-mcp": {{
      "command": "uvx",
      "args": ["douyin-mcp-server"],
      "env": {{
        "ASR_PROVIDER": "volcengine",
        "ARK_API_KEY": "your-ark-key",
        "ARK_BASE_URL": "{DEFAULT_ARK_BASE_URL}",
        "ARK_ASR_MODEL": "{DEFAULT_ARK_MODEL}"
      }}
    }}
  }}
}}
```

## 工具说明
- `extract_douyin_text`: 完整文本提取流程（需要配置 ASR 提供方）
- `get_douyin_download_link`: 获取无水印视频下载链接（无需 API 密钥）
- `parse_douyin_video_info`: 仅解析视频基本信息
- `douyin://video/{{video_id}}`: 获取指定视频的详细信息
"""


def main():
    """启动 MCP 服务器"""
    mcp.run()


if __name__ == "__main__":
    main()
