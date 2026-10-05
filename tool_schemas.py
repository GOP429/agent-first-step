TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "count_todos",
            "description": "统计当前待办数量",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "添加一条待办事项",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "需要添加的待办内容",
                    },
                },
                "required": ["content"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式",
                    },
                },
                "required": ["expression"],
            },
        },
    },
     {
            "type": "function",
            "function": {
                "name": "list_todos",
                "description": "查看全部待办事项",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },

]