from langgraph.checkpoint.sqlite import SqliteSaver

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from pathlib import Path

from langchain.tools import tool

from tools import TodoStore, safe_calculate


data_path = Path(__file__).parent / "data" / "todos_langchain.json"
memory_path = (
    Path(__file__).parent
    / "data"
    / "agent_memory.sqlite"
)
todo_store = TodoStore(data_path)


@tool
def count_todos() -> str:
    """统计当前待办事项的数量。"""
    return todo_store.count()

@tool
def add_todo(content: str) -> str:
    """添加一条新的待办事项。"""
    return todo_store.add(content)


@tool
def list_todos() -> str:
    """查看当前全部待办事项。"""
    return todo_store.list_all()


@tool
def calculate(expression: str) -> str:
    """计算一个基础数学算式。"""
    return safe_calculate(expression)

def build_agent(checkpointer=None):
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
    )

    return create_agent(
        model=model,
        tools=[
            add_todo,
            list_todos,
            count_todos,
            calculate,
        ],
        system_prompt="你是一个待办事项助理。需要操作待办或计算时，必须使用提供的工具。",
        checkpointer=checkpointer,
    )
def main():
    memory_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SqliteSaver.from_conn_string(
        str(memory_path)
    ) as checkpointer:
        agent = build_agent(
            checkpointer=checkpointer,
        )
        
        thread_id = input(
            "请输入会话编号："
        ).strip()

        if not thread_id:
            thread_id = "default-session"

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        while True:
            user_input = input("你：").strip()

            if user_input in {"退出", "exit", "quit"}:
                break

            if not user_input:
                continue

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config,
            )

            print(
                f"Agent："
                f"{result['messages'][-1].content}"
            )
if __name__ == "__main__":
    main()