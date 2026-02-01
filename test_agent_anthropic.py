import asyncio
import os

from anthropic import AsyncAnthropic
from openreward import AsyncOpenReward

MODEL_NAME = "claude-sonnet-4"
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Still needed for grading


async def main() -> None:
    or_client = AsyncOpenReward()
    anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    environment = or_client.environments.get(
        name="local/RubricHub",
        base_url="http://localhost:8080"
    )

    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="anthropic")

    print(f"Found {len(tasks)} tasks")
    print(f"Testing first task: {tasks[0]}")

    task = tasks[0]

    async with environment.session(
        task=task,
        secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        prompt = await session.get_prompt()
        prompt_text = prompt if isinstance(prompt, str) else prompt[0].text

        messages = [{"role": "user", "content": prompt_text}]
        finished = False

        while not finished:
            response = await anthropic_client.messages.create(
                model=MODEL_NAME,
                max_tokens=4096,
                tools=tools,
                messages=messages
            )

            # Process tool calls
            for content in response.content:
                if content.type == "tool_use":
                    tool_result = await session.call_tool(
                        content.name,
                        content.input
                    )

                    finished = tool_result.finished
                    reward = tool_result.reward

                    print(f"\nTool: {content.name}")
                    print(f"Reward: {reward:.3f}")
                    print(f"\nFeedback:\n{tool_result.blocks[0].text}")

                    if finished:
                        print("\nEpisode finished!")
                        break

            if response.stop_reason != "tool_use":
                break


if __name__ == "__main__":
    asyncio.run(main())
