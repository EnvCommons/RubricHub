import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


async def main() -> None:
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    environment = or_client.environments.get(
        name="local/RubricHub",
        base_url="http://localhost:8080"
    )

    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")
    print(f"Testing first task: {tasks[0]}")

    task = tasks[0]

    async with environment.session(
        task=task,
        secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        prompt = await session.get_prompt()
        prompt_text = prompt if isinstance(prompt, str) else prompt[0].text + "\n\n" + "Always use the submit_response tool to submit your response."

        input_list = [{"role": "user", "content": prompt_text}]
        finished = False

        while not finished:
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            input_list += response.output
            print(input_list[-1])

            for item in response.output:
                if item.type == "function_call":
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments)),
                    )

                    finished = tool_result.finished
                    reward = tool_result.reward

                    print(f"\nTool: {item.name}")
                    print(f"Reward: {reward:.3f}")
                    print(f"\nFeedback:\n{tool_result.blocks[0].text}")

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result.blocks[0].text
                    })
                    print(input_list[-1])

                    if finished:
                        print("\nEpisode finished!")
                        break

            if not any(i.type == "function_call" for i in response.output):
                break


if __name__ == "__main__":
    asyncio.run(main())
