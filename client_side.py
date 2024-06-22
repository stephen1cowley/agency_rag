import requests
import random
from langchain_core.messages import AIMessage, HumanMessage

query = "What's the weather like today?"

payload = {
    "input":{
        "messages":[{
            "content":"What is the latest UK politics news headline and its relation to the 1997 election?","type":"human"
        }],
        "is_last_step":False
    },
    "config":{
        "configurable":{
            "thread_id":f"thread1748{random.randint(1,1000)}"
        }
    }
}

response = requests.post(
    "http://localhost:8000/my_runnable/invoke",
    json=payload
)
print(response.json())
