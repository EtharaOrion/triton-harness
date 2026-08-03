from openai import OpenAI
import json
from anthropic import AnthropicVertex

class BaseAIClient():
    def __init__(self) -> None:
        pass

    def inference(self, messages, n):
        raise NotImplementedError("Please Use the correct client, This is the base class!")

class OpenAIClient(BaseAIClient):
    """
    you can use this api as openai client or local model deployed on your own machine with vllm, ollama, etc.
    or any online model with compatible api, for example, deepseek.
    """
    def __init__(self, url, key, model) -> None:
        
        self.client = OpenAI(
            base_url=url,
            api_key=key,
        )
        self.model = model

    def inference(self, messgaes, n):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messgaes,
            n=n
        )
        return [completion.choices[i].message.content.strip() for i in range(n)]
    

class VertextAIClient(BaseAIClient):
    """
    this api is for anthropic vertex ai, please refer to the official website for more details.
    """
    def __init(self, region, project_id, model_name):
        self.region = region
        self.project_id = project_id
        self.model_name = model_name
        self.client = AnthropicVertex(region=region, project_id=project_id)

    def infrence(self, messages, n):

        message = self.client.messages.create(
            max_tokens=1024,
            temperature=0.5,
            messages=messages,
            model=self.model_name,
        )
        return [message.content[0].text]


class AnthropicBridgeClient(BaseAIClient):
    """Talks to the Claude Code OAuth bridge (../../proxy/) via the native
    Messages API. The bridge only injects OAuth + the required system prefix on
    /v1/messages, so the OpenAI ChatCompletions path can't be used. `url` is the
    bridge root (e.g. http://127.0.0.1:8765); `key` is a stub the bridge replaces.
    """
    def __init__(self, url, key, model, max_tokens=4096) -> None:
        from anthropic import Anthropic
        self.client = Anthropic(
            base_url=url,
            api_key=key or "sk-ant-oauth-bridge-stub",
            max_retries=5,
        )
        self.model = model
        self.max_tokens = max_tokens

    def _shrink(self, messages):
        shrunk = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                m = {**m, "content": content[:int(len(content) * 0.65)]}
            shrunk.append(m)
        return shrunk

    def _create_with_shrink(self, messages):
        for attempt in range(6):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )
            except Exception as e:
                if "prompt is too long" not in str(e).lower() or attempt == 5:
                    raise
                messages = self._shrink(messages)
                print(f"prompt is too long, retrying with truncated context (attempt {attempt + 1})")

    def inference(self, messages, n):
        # The Messages API has no `n` param; sample sequentially to match the
        # BaseAIClient contract (return a list of n completions).
        results = []
        for _ in range(n):
            message = self._create_with_shrink(messages)
            text = "".join(
                block.text for block in message.content
                if getattr(block, "type", None) == "text"
            )
            results.append(text.strip())
        return results

