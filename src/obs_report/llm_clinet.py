# src/obs_report/llm_client.py

from typing import List, Optional, Tuple
import requests
from pydantic import BaseModel, Field
from .parser import DocumentData
from pathlib import Path
import re
from abc import ABC, abstractmethod
import json
from functools import wraps
from typing import Callable, TypeVar, Any
from .parser import parse_markdown
from .watcher import get_recent_changes
import ollama
import os

class LLMRequest(BaseModel):
    model: str = Field(..., description="LLM model name")
    messages: List[dict] = Field(..., description="List of messages in chat completion format")
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens")

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: Message
    finish_reason: str
    index: int

class LLMResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]

class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434/v1/chat/completions", language: str = "Korean"):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = 0.1
        self.max_tokens: Optional[int] = 1024
        self.system_prompt = f"You are a helpful assistant. Always answer in {language}."
        self.timeout = 120  # Increased timeout to 120 seconds
        self.max_retries = 3  # Maximum number of retries
        self.chat_history = {}  # 대화 기록 저장
        self.current_attempt = 0

    def __copy__(self):
        """얕은 복사 구현"""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """깊은 복사 구현"""
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def clone(self) -> 'BaseLLMClient':
        """깊은 복사를 수행하는 편의 메서드"""
        import copy
        return copy.deepcopy(self)

    def set_system_prompt(self, prompt: str, mode : str = 'append') -> None:
        """Set the system prompt"""
        if mode == 'append':
            self.system_prompt = self.system_prompt + "\n" + prompt
        elif mode == 'replace':
            self.system_prompt = prompt
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _create_request(self, messages: List[dict]) -> LLMRequest:
        """Create LLM request object"""
        # Add system prompt only if not present
        messages = [{"role": "system", "content": self.system_prompt}] + messages

        # 대화 기록에 새로운 메시지 추가
        self.chat_history.extend(messages)

        return LLMRequest(
            model=self.model_name,
            messages=self.chat_history,  # 전체 대화 기록 사용
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _call_llm(self, request: LLMRequest) -> LLMResponse:
        """Call LLM API (with retry logic)"""
        payload = request.model_dump()
        
        # max_tokens가 None이면 해당 필드 제거
        if payload.get('max_tokens') is None:
            payload.pop('max_tokens', None)
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.base_url, 
                    json=payload, 
                    timeout=self.timeout
                )
                resp.raise_for_status()
                return LLMResponse(**resp.json())
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:  # If this was the last attempt
                    raise  # Re-raise the exception
                print(f"Request timeout. Retrying ({attempt + 1}/{self.max_retries})...")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {str(e)}")
                raise

    @abstractmethod
    def process_response(self, response: LLMResponse) -> str:
        """Process response (to be implemented by subclasses)"""
        pass

    def chat(self, messages: List[dict]) -> str:
        """Send chat request and process response"""
        request = self._create_request(messages)
        response = self._call_llm(request)
        report = self.process_response(response)

        history = messages + [{"role": "assistant", "content": report}]

        self.chat_history[self.current_attempt] = history
        self.current_attempt += 1
        return report
    
    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = {}
        self.current_attempt = 0

class DeepSeekClient(BaseLLMClient):
    """Client for DeepSeek model"""

    def __init__(self):
        super().__init__(model_name="deepseek-r1:14b")

    def process_response(self, response: LLMResponse) -> str:
        """Remove <think> blocks from DeepSeek response"""
        content = response.choices[0].message.content
        return self._remove_think_blocks(content)

    def _remove_think_blocks(self, text: str) -> str:
        """Remove <think> blocks and clean up text"""
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Combine consecutive empty lines into one
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

class Phi4Client(BaseLLMClient):
    """Client for Phi4 model"""
    
    def __init__(self):
        super().__init__(model_name="phi4:latest")

    def process_response(self, response: LLMResponse) -> str:
        """Remove <think> blocks from Phi4 response and return the content"""
        content = response.choices[0].message.content
        return content

class GeneratorAgent:
    """Report generation agent"""
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.llm_client.set_system_prompt(
"""You are a helpful assistant specialized in generating two types of reports:

1) Report Type = NoteSummary  
   - Generate a concise summary for a single note.

2) Report Type = DailyReport  
   - Generate an aggregated daily activity report based on multiple NoteSummary outputs that you receive as input.

Common Rules (apply to both report types):  
- Always produce structured output using bullet points.  
- Keep language clear and concise; avoid unnecessary explanations.  
- If a section has no content, omit it entirely.  
- Answer in Markdown format.

---

NoteSummary Rules (when Report Type = NoteSummary):  
0. **Empty Note Exception**  
   - If the note’s content is empty or contains only whitespace, output exactly:  
     ```
     This note is empty. No summary available.
     ```  
   - Do not produce Keywords, Work Type, or Summary sections.

1. **Determine Note Type**  
   - **Created Note**: summarize the full file content.  
   - **Modified Note**: summarize only the listed changes (refer to the current content for full context); do not restate the full content.  
   - **Deleted Note**: summarize the content of the deleted file.

2. For all non-empty notes, use up to 5 sentences total, organized into these bullet sections in order:  
   - **Keywords**: key terms or phrases extracted from the relevant content (full content for created, change snippets for modified, deleted content for deleted)  
   - **Work Type**: one of Research, Meeting Notes, Idea Organization, etc.  
   - **Summary**: concise main points (for modified notes, describe what changed; for deleted notes, describe what was in the file)

3. Use bullet points for each section.

---

DailyReport Rules (when Report Type = DailyReport):  
- You will receive a list of filenames and their corresponding NoteSummary outputs.  
- **Do not** reprint the full NoteSummary contents.  
- Under three headings, list only the filenames:  
  * Newly Created Notes  
  * Modified Notes  
  * Deleted Notes  
- **Individual File Summaries**: For each filename, provide a brief summary based on its NoteSummary.  
- **Comprehensive Activity Report**: Based on all NoteSummary outputs, write a cohesive report that covers:  
  * **Overall Work Types**: frequency/counts of each work type  
  * **Key Themes**: main topics or keywords across all notes  
  * **Activity Patterns**: brief observations on the day’s work distribution

---

When you receive a request, you will be told which report type to use and provided the appropriate input (raw note data for NoteSummary or a filename list with NoteSummary outputs for DailyReport). Follow only the rules for that report type, and produce output that strictly adheres to this structure.
"""
        , mode='append')

    def summarize_file_created(self, doc: DocumentData) -> str:
        """Summarize single file content"""
        file_name = Path(doc.path).name
        messages = [{
            "role": "user",
            "content": f"Generate a NoteSummary base on the information below:\n\n"
                      f"Filename: {file_name}\n"
                      f"Report Type: NoteSummary\n"
                      f"Note Type: Created Note\n"
                      f"Title: {doc.title}\n"
                      f"Headers: {', '.join(doc.headers)}\n"
                      f"Content: {doc.content}\n"
        }]
        return self.llm_client.chat(messages)

    def summarize_file_changes(self, doc: DocumentData) -> str:
        """Summarize file changes"""
        file_name = Path(doc.path).name
        changes_text = "\n".join(doc.changes) if doc.changes is not None else "변경사항 없음"
        messages = [{
            "role": "user",
            "content": f"Generate a NoteSummary base on the information below':\n\n"
                      f"Filename: {file_name}\n"
                      f"Report Type: NoteSummary\n"
                      f"Note Type: Modified Note\n"
                      f"Title: {doc.title}\n"
                      f"Headers: {', '.join(doc.headers)}\n"
                      f"Current Content: {doc.content}\n"
                      f"Changes:\n{changes_text}"
        }]
        return self.llm_client.chat(messages)

    def summarize_deleted_file(self, doc: DocumentData) -> str:
        """Summarize deleted file (based on filename)"""
        file_name = Path(doc.path).name
        messages = [{
            "role": "user",
            "content": f"Generate a NoteSummary base on the information below:\n\n"
                      f"Filename: {file_name}\n"
                      f"Report Type: NoteSummary\n"
                      f"Note Type: Deleted Note\n"
                      f"Title: {doc.title}\n"
                      f"Headers: {', '.join(doc.headers)}\n"
                      f"Content: {doc.content}\n"
        }]
        return self.llm_client.chat(messages)

class CriticAgent:
    """Report validation agent"""
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.llm_client.set_system_prompt(
            """
You are a strict Quality Control Agent for note summaries.  
Your job is to check whether a generated report fully complies with its system prompt and the user's original request.

Inputs (in this exact order):  
1. The System Prompt used by the report-generating agent.  
2. The User Request given to the report-generating agent
3. The Report text produced by that agent.  

Tasks:  
- Validate that every requirement in the System Prompt is satisfied.  
- Confirm that the Report also correctly follows the User Request (e.g., correct report_type and note_type handling).  
- Identify any violations, omissions, or formatting errors.

Output:  
Return a single JSON object with three fields:
```json
{
"is_valid": <true|false>,
"issues":    [<simple, concise list of violations>],
"suggestions":[<simple, concise list of fixes>]
}
"""
        , mode='append')

    def validate_report(self, report: str, target_prompt: str, user_request: str) -> Tuple[bool, List[str], List[str]]:
        """Validate report"""
        messages = [{
            "role": "user",
            "content": f"Validate the following report:\n\n"
                      f"Target LLM's System Prompt:\n{target_prompt}\n\n"
                      f"Target LLM's User Request:\n{user_request}\n\n"
                      f"Generated Report:\n{report}"
        }]
        response = self.llm_client.chat(messages)
        
        # Try to parse JSON
        try:
            # Extract JSON part from response
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            validation = json.loads(json_str)
            return (
                validation.get("is_valid", False),
                validation.get("issues", []),
                validation.get("suggestions", [])
            )
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Original response: {response}")
            return False, ["Could not parse validation result."], []

T = TypeVar('T')

def with_validation(max_attempts: int = 3):
    """
    리포트 생성 함수에 검증 로직을 추가하는 데코레이터
    
    Args:
        max_attempts: 최대 재시도 횟수
    """
    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        @wraps(func)
        def wrapper(self: 'NoteManager', *args: Any, **kwargs: Any) -> str:
            # 초기 리포트 생성
            report = func(self, *args, **kwargs)
            assistant_report_list = []
            
            # 검증 및 개선 반복
            for attempt in range(max_attempts):
                user_request = '\n'.join([f"{msg['content']}" for msg in self.generator.llm_client.chat_history[attempt] if msg['role'] == 'user'])
                assistant_report = '\n'.join([f"{msg['content']}" for msg in self.generator.llm_client.chat_history[attempt] if msg['role'] == 'assistant'])
                assistant_report_list.append(assistant_report)
                is_valid, issues, suggestions = self.critic.validate_report(
                    report, 
                    self.generator.llm_client.system_prompt,
                    user_request
                )
                
                if is_valid:
                    # 검증 성공 시 대화 내용 초기화
                    self.generator.llm_client.clear_chat_history()
                    self.critic.llm_client.clear_chat_history()
                    return report, is_valid
                
                if attempt < max_attempts - 1:
                    past_reports = ''
                    for i, assistant_report in enumerate(assistant_report_list):
                        past_reports += f"이전 시도 {i+1}의 내용:\n{assistant_report}\n"

                    # 개선된 리포트 생성 (이전 리포트 내용 포함)
                    report = self.generator.llm_client.chat([{
                        "role": "user",
                        "content": f"이전 리포트에 다음과 같은 문제가 있었습니다. 해당 내용을 바탕으로 기존 리포트를 수정해주세요. 수정 이후의 리포트만을 반환해주세요.:\n" + 
                                  "\n".join(f"- {issue}" for issue in issues) + "\n\n" +
                                  "개선 제안:\n" +
                                  "\n".join(f"- {suggestion}" for suggestion in suggestions) + "\n\n" +
                                  "이전 시도들의 내용:\n" + past_reports
                    }])
                else:
                    print("\n최대 시도 횟수 도달. 마지막 리포트를 반환합니다.")
                    # 최대 시도 횟수 도달 시에도 대화 내용 초기화
                    self.generator.llm_client.clear_chat_history()
                    self.critic.llm_client.clear_chat_history()

            return report, is_valid
        return wrapper
    return decorator

class NoteManager:
    """노트 관리 및 요약 생성을 담당하는 클래스"""
    
    def __init__(self, generator: GeneratorAgent, critic: CriticAgent):
        self.generator = generator
        self.critic = critic
    
    @with_validation()
    def _summarize_file(self, doc: DocumentData) -> str:
        """단일 파일 요약 생성 및 검증"""
        return self.generator.summarize_file_created(doc)
    
    @with_validation()
    def _summarize_file_changes(self, doc: DocumentData) -> str:
        """파일 변경사항 요약 생성 및 검증"""
        return self.generator.summarize_file_changes(doc)
    
    @with_validation()
    def _summarize_deleted_file(self, doc: DocumentData) -> str:
        """삭제된 파일 요약 생성 및 검증"""
        return self.generator.summarize_deleted_file(doc)
    
    @with_validation()
    def _generate_daily_report(
        self,
        created_files: List[DocumentData],
        modified_files: List[DocumentData],
        deleted_files: List[DocumentData]
    ) -> str:
        """Generate daily activity report"""
        summaries_created = [self._summarize_file(doc) for doc in created_files] #List[Tuple[content, is_valid]]
        summaries_modified = [self._summarize_file_changes(doc) for doc in modified_files] #List[Tuple[content, is_valid]]
        summaries_deleted = [self._summarize_deleted_file(doc) for doc in deleted_files] #List[Tuple[content, is_valid]]

        valid_created = [summary for summary, is_valid in summaries_created if is_valid]
        valid_modified = [summary for summary, is_valid in summaries_modified if is_valid]
        valid_deleted = [summary for summary, is_valid in summaries_deleted if is_valid]

        # 각 섹션별 내용 구성
        sections = []
        
        if created_files:
            sections.append(
                "1. **새롭게 생성된 노트:**\n\n" +
                "\n".join(f"- **노트 제목:** {doc.title}\n{summary}" 
                         for doc, summary in zip(created_files, valid_created))
            )
        else:
            sections.append("1. **새롭게 생성된 노트:**\n\n" + "없음")
        
        if modified_files:
            sections.append(
                "2. **수정된 노트:**\n\n" +
                "\n".join(f"- **노트 제목:** {doc.title}\n{summary}" 
                         for doc, summary in zip(modified_files, valid_modified))
            )
        else:
            sections.append("2. **수정된 노트:**\n\n" + "없음")
        
        if deleted_files:
            sections.append(
                "3. **삭제된 노트:**\n\n" +
                "\n".join(f"- **노트 제목:** {doc.title}\n{summary}" 
                         for doc, summary in zip(deleted_files, valid_deleted))
            )
        else:
            sections.append("3. **삭제된 노트:**\n\n" + "없음")

        messages = [{
            "role": "user",
            "content": (
            f"Generate a DailyReport base on the information below:\n\n"
            f"Report Type: DailyReport\n"
            f"Newly Created Notes:\n{'\n'.join(sections[0])}\n"
            f"Modified Notes:\n{'\n'.join(sections[1])}\n"
            f"Deleted Notes:\n{'\n'.join(sections[2])}\n"
            )
        }]

        report = self.generator.llm_client.chat(messages)
        return report
    
    def write_daily_report(self, dirpath: Path, created_files, modified_files, deleted_files) -> None:
        """Write daily report to file"""
        # 오늘 날짜 구하기
        date_str = __import__("datetime").date.today().isoformat()
        report, is_valid = self._generate_daily_report(created_files, modified_files, deleted_files)
        
        # 파일 경로 생성
        report_path = dirpath / f"{date_str}.md"
        
        # 파일 저장
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Daily report written to {report_path}")

if __name__ == "__main__":
    import os
    from pathlib import Path
    from parser import parse_markdown
    from watcher import get_recent_changes
    
    # 테스트 실행
    vault_dir = Path.home() / "Documents" / "Obsidian Vault"
    ignore_dirs = os.path.join(vault_dir,"Calendar", "01_Daily")
    created, deleted, modified = get_recent_changes(vault_dir, exclude_dirs=[ignore_dirs], update_state=False)
    test_dir = ignore_dirs
    os.makedirs(test_dir, exist_ok=True)

    print("====================================")
    print("생성된 파일:")
    print(created)
    print("====================================")
    print("삭제된 파일:")
    print(deleted)
    print("====================================")
    print("수정된 파일:")
    print(modified)
    print("====================================")
    
    # 파일 파싱
    created_docs = [parse_markdown(Path(p), content=created[p]) for p in created]
    modified_docs = [parse_markdown(Path(p), changes=changes) for p, changes in modified.items()]
    deleted_docs = [parse_markdown(Path(p), content=deleted[p]) for p in deleted]
    
    # LLM 클라이언트 초기화
    llm_client = Phi4Client()
    generator = GeneratorAgent(llm_client.clone())
    critic = CriticAgent(llm_client.clone())

    print("====================================")
    print(generator.llm_client.system_prompt)
    print("====================================")
    print(critic.llm_client.system_prompt)
    print("====================================")
    
    # NoteManager 초기화 및 실행
    manager = NoteManager(generator, critic)
    manager.write_daily_report(test_dir, created_docs, modified_docs, deleted_docs)

