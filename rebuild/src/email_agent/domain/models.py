# 这个文件定义模型
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv(".env", override=True)

model = ChatDeepSeek(
    model="deepseek-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)