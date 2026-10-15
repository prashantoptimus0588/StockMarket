import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main():
    url = os.environ["MCP_SERVER_URL"]
    print("Testing MCP:", url, flush=True)

    client = MultiServerMCPClient({
        "server": {
            "url": url,
            "transport": "streamable_http",
        }
    })

    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=60)
        print("Tools discovered:", len(tools))
        for tool in tools:
            print("-", tool.name)
    except Exception as exc:
        print("Error type:", type(exc).__name__)
        print("Error:", repr(exc))


if __name__ == "__main__":
    asyncio.run(main())