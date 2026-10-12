"""
LLM 可插拔层
============
统一接口，通过环境变量 LLM_PROVIDER 切换后端：
  - ollama（默认）: 本地 Ollama 服务
  - cloud        : 云端 OpenAI 兼容 API（如 DeepSeek）

用法：
    from llm import get_llm
    llm = get_llm()
    answer = llm.chat([{"role": "user", "content": "你好"}])
"""
import os
from typing import List, Dict

import requests

from config import (
    LLM_PROVIDER, OLLAMA_URL, OLLAMA_MODEL,
    CLOUD_BASE_URL, CLOUD_MODEL, CLOUD_API_KEY,
    TEMPERATURE, MAX_TOKENS,
)


class BaseLLM:
    """LLM 统一接口"""

    name = "base"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """messages: [{"role": "user"/"system"/"assistant", "content": "..."}]"""
        raise NotImplementedError

    def is_available(self) -> bool:
        """检查后端是否可用"""
        raise NotImplementedError


class MockLLM(BaseLLM):
    """
    本地零依赖的"假"大模型（离线兜底）
    ----------------------------------
    不调用任何模型，直接把检索到的评论整理成答案。
    用途：Ollama / 云端都不可用时，仍能验证 RAG 的"检索"环节。

    ⚠️ 它不是真正的生成，只是把评论拼起来 —— 用于开发调试与演示，
       不能体现大模型的归纳/润色能力。
    """

    name = "mock"

    def is_available(self) -> bool:
        return True

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # 从 user 消息里把评论原文抠出来，做个朴素整理
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        lines = [l for l in user_msg.split("\n") if l.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))]
        if not lines:
            return "（离线模式）未找到相关评论。"

        out = ["（离线模式：以下为检索到的相关评论原文，未经过大模型归纳）\n"]
        for l in lines:
            out.append(f"  {l.strip()}")
        out.append("\n提示：启动 Ollama 或配置云端 API Key 后，此处会由大模型生成归纳性回答。")
        return "\n".join(out)


class OllamaLLM(BaseLLM):
    """本地 Ollama（零成本、离线、集显可跑）"""

    name = "ollama"

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
            },
            timeout=180,   # 本地模型首次加载慢，给足时间
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


class CloudLLM(BaseLLM):
    """云端 API（OpenAI 兼容接口，效果更好）"""

    name = "cloud"

    def __init__(self, model: str = CLOUD_MODEL, base_url: str = CLOUD_BASE_URL,
                 api_key: str = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or CLOUD_API_KEY

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError("未配置云端 API Key（环境变量 DEEPSEEK_API_KEY）")
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def get_llm(provider: str = None) -> BaseLLM:
    """
    工厂函数：根据配置返回 LLM 实例。

    优先级：
      provider 指定 → 该后端可用就用它
                    → 不可用则依次尝试：ollama → cloud → mock
    最终一定返回一个可用实例（mock 永远可用，作为离线兜底）。
    """
    p = (provider or LLM_PROVIDER).lower()

    # 用户显式指定 mock
    if p == "mock":
        return MockLLM()

    # 按优先级排列候选
    if p == "cloud":
        candidates = [CloudLLM(), OllamaLLM(), MockLLM()]
    else:  # ollama（默认）
        candidates = [OllamaLLM(), CloudLLM(), MockLLM()]

    for i, c in enumerate(candidates):
        if c.is_available():
            if i > 0:
                print(f"[提示] {candidates[0].name} 不可用，已回退到 {c.name}")
            return c

    return MockLLM()
