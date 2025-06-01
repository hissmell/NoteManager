import ollama
import sys
import subprocess
import time
import os
from contextlib import contextmanager

@contextmanager
def llm_manager(model_name: str = "deepseek-r1:14b"):
    """LLM을 시작하고 종료하는 컨텍스트 매니저"""
    # 서버 시작
    subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
    
    # 서버가 시작될 때까지 잠시 대기
    time.sleep(2)
    
    try:
        yield
    finally:
        subprocess.Popen(['ollama', 'stop', model_name], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

# Ollama 클라이언트 설정
client = ollama.Client(host='http://localhost:11434')

# with 구문으로 서버 관리
with llm_manager(model_name="deepseek-r1:14b"):
    response = client.chat(
        model='deepseek-r1:14b',
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    print(response['message']['content'])

