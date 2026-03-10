#!/usr/bin/env python3
"""
抖音视频文案提取器 WebUI

启动方式:
    cd douyin-mcp-server
    export ASR_PROVIDER=dashscope
    export API_KEY="sk-xxx"
    python web/app.py
    # 访问 http://localhost:8080
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "douyin-video" / "scripts"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import uvicorn

from douyin_downloader import HEADERS, extract_text, get_video_info, resolve_asr_config

app = FastAPI(title="抖音文案提取器", version="1.1.0")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


class VideoRequest(BaseModel):
    url: str
    api_key: str = ""
    provider: str = ""
    api_base_url: str = ""
    model: str = ""


class VideoInfoResponse(BaseModel):
    success: bool
    video_id: str = ""
    title: str = ""
    download_url: str = ""
    error: str = ""


class ExtractResponse(BaseModel):
    success: bool
    video_id: str = ""
    title: str = ""
    text: str = ""
    download_url: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    config = resolve_asr_config()
    return {
        "status": "ok",
        "api_key_configured": bool(config["api_key"]),
        "provider": config["provider"],
        "model": config["model"],
        "api_base_url": config["api_base_url"],
    }


@app.post("/api/video/info", response_model=VideoInfoResponse)
async def get_info(req: VideoRequest):
    try:
        info = get_video_info(req.url)
        return VideoInfoResponse(success=True, video_id=info["video_id"], title=info["title"], download_url=info["url"])
    except Exception as e:
        return VideoInfoResponse(success=False, error=str(e))


@app.post("/api/video/extract", response_model=ExtractResponse)
async def extract_transcript(req: VideoRequest):
    provider = req.provider or None
    api_base_url = req.api_base_url or None
    model = req.model or None

    try:
        result = extract_text(
            req.url,
            api_key=req.api_key or None,
            show_progress=False,
            provider=provider,
            api_base_url=api_base_url,
            model=model,
        )
        return ExtractResponse(
            success=True,
            video_id=result["video_info"]["video_id"],
            title=result["video_info"]["title"],
            text=result["text"],
            download_url=result["video_info"]["url"],
            provider=result["provider"],
            model=result["model"],
        )
    except Exception as e:
        return ExtractResponse(success=False, error=str(e))


@app.get("/api/video/download")
async def download_video(url: str, filename: str = "video.mp4"):
    try:
        download_headers = {
            **HEADERS,
            "Referer": "https://www.douyin.com/",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive",
        }
        response = requests.get(url, headers=download_headers, stream=True, allow_redirects=True)
        response.raise_for_status()
        content_length = response.headers.get("content-length", "")

        def iter_content():
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        if content_length:
            headers["Content-Length"] = content_length

        return StreamingResponse(iter_content(), media_type="video/mp4", headers=headers)
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"下载失败: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    port = int(os.getenv("PORT", "8080"))
    config = resolve_asr_config()
    print(f"🚀 启动文案提取器 WebUI: http://localhost:{port}")
    print(f"🧠 默认 Provider: {config['provider']}")
    print(f"🔑 API 配置状态: {'已配置' if config['api_key'] else '未配置'}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
