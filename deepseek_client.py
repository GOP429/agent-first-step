import requests

class DeepSeekAPIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    def create_chat_completion(self, messages, tools):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
            },
            timeout=30,
        )

        response.raise_for_status()
        return response.json()