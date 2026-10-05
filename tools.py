"""The deterministic tools that our first mini-agent is allowed to use."""

from __future__ import annotations

import ast
import json
import operator
import re
import unicodedata
from pathlib import Path


class TodoStore:
    """A tiny JSON-backed todo store."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> list[str]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, todos: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(todos, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("待办内容不能为空")

        todos = self._load()
        todos.append(content)
        self._save(todos)
        return f"已添加待办 #{len(todos)}：{content}"

    def list_all(self) -> str:
        todos = self._load()
        if not todos:
            return "目前没有待办。"
        return "\n".join(f"{index}. {item}" for index, item in enumerate(todos, 1))
    def count(self) -> str:
        todos = self._load()
        return f"当前待办数量为 {len(todos)}"


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> str:
    """Evaluate basic arithmetic without using Python's unsafe eval()."""

    # Users often enter full-width Chinese punctuation or Markdown-style escapes.
    # NFKC converts characters such as （１２＋８） into their ASCII equivalents.
    expression = unicodedata.normalize("NFKC", expression).strip()
    expression = expression.replace("×", "*").replace("÷", "/").replace("−", "-")
    expression = re.sub(r"\\([+\-*/%()])", r"\1", expression)
    if not expression or len(expression) > 100:
        raise ValueError("请输入长度不超过 100 个字符的算式")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("算式中只能包含数字")
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("指数绝对值不能超过 10")
            return _BINARY_OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))

        raise ValueError("只支持数字、括号和 + - * / % ** 运算")

    result = evaluate(tree)
    return f"计算结果：{result}"
