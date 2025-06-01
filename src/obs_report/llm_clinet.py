# src/obs_report/llm_client.py

from typing import List, Optional, Tuple
import requests
from pydantic import BaseModel, Field
from parser import DocumentData
from pathlib import Path
import re
from abc import ABC, abstractmethod
import json

class LLMRequest(BaseModel):
    model: str = Field(..., description="LLM model name")
    messages: List[dict] = Field(..., description="List of messages in chat completion format")
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, description="Maximum number of tokens")

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
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434/v1/chat/completions"):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = 0.1
        self.max_tokens = 1024
        self.system_prompt = "You are a helpful assistant."
        self.timeout = 120  # Increased timeout to 120 seconds
        self.max_retries = 3  # Maximum number of retries

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt"""
        self.system_prompt = prompt

    def _create_request(self, messages: List[dict]) -> LLMRequest:
        """Create LLM request object"""
        # Add system prompt only if not present
        if not any(msg.get("role") == "system" for msg in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        return LLMRequest(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _call_llm(self, request: LLMRequest) -> LLMResponse:
        """Call LLM API (with retry logic)"""
        payload = request.model_dump()
        
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
        return self.process_response(response)

class DeepSeekClient(BaseLLMClient):
    """Client for DeepSeek model"""

    def __init__(self):
        super().__init__(model_name="deepseek-r1:14b")
        self.set_system_prompt(
            "You are a helpful assistant that summarizes Obsidian notes. "
            "You should focus on extracting key information and providing concise summaries. "
            "Always respond in English."
        )

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

class GeneratorAgent(DeepSeekClient):
    """Report generation agent"""
    
    def __init__(self):
        super().__init__()
        self.set_system_prompt(
            "You are a helpful assistant that summarizes Obsidian notes. "
            "You should follow these rules strictly:\n"
            "1. Always respond in English\n"
            "2. Focus on extracting key information and providing concise summaries\n"
            "3. Include the following sections in your response:\n"
            "   - Key keywords for each note\n"
            "   - Type of work performed (e.g., research, idea organization, meeting notes)\n"
            "   - Main content summary\n"
            "4. Keep each note's summary up to 4 sentences\n"
            "5. If the note is empty, return 'Empty note'\n"
            "6. Use bullet points for better readability\n"
            "7. Write the summary in the markdown format"
        )

    def summarize_file_content(self, doc: DocumentData) -> str:
        """Summarize single file content"""
        messages = [{
            "role": "user",
            "content": f"Please summarize the following Obsidian note:\n\n"
                      f"Title: {doc.title}\n"
                      f"Headers: {', '.join(doc.headers)}\n"
                      f"Content: {doc.content[:2000]}..."  # Send only first 2000 characters
        }]
        return self.chat(messages)

    def summarize_file_changes(self, doc: DocumentData, changes: List[str]) -> str:
        """Summarize file changes"""
        messages = [{
            "role": "user",
            "content": f"Please summarize the changes in the following Obsidian note:\n\n"
                      f"Title: {doc.title}\n"
                      f"Changes:\n{''.join(changes)}"
        }]
        return self.chat(messages)

    def summarize_deleted_file(self, file_path: str) -> str:
        """Summarize deleted file (based on filename)"""
        file_name = Path(file_path).stem
        messages = [{
            "role": "user",
            "content": f"The following Obsidian note has been deleted. Based on the filename, please summarize what this note might have contained:\n\n"
                      f"Filename: {file_name}\n\n"
                      "Based on the filename, please summarize:\n"
                      "1. What topic this note might have covered\n"
                      "2. What kind of work might have been done\n"
                      "3. Why this note might have been deleted"
        }]
        return self.chat(messages)

    def generate_daily_report(
        self,
        created_files: List[Tuple[DocumentData, str]],  # (document, summary)
        modified_files: List[Tuple[DocumentData, str]],  # (document, change summary)
        deleted_files: List[Tuple[str, str]]    # (file path, summary)
    ) -> str:
        """Generate daily activity report"""
        date_str = __import__("datetime").date.today().isoformat()
        
        messages = [{
            "role": "user",
            "content": f"Please provide a comprehensive summary of today's ({date_str}) Obsidian note activities.\n\n"
                      f"1. Newly created notes:\n" + 
                      "\n".join(f"- {doc.title}\n{summary}" for doc, summary in created_files) + "\n\n" +
                      f"2. Modified notes:\n" +
                      "\n".join(f"- {doc.title}\n{summary}" for doc, summary in modified_files) + "\n\n" +
                      f"3. Deleted notes:\n" +
                      "\n".join(f"- {Path(path).name}\n{summary}" for path, summary in deleted_files) + "\n\n" +
                      "Based on the above information, please provide a comprehensive summary of today's Obsidian note activities. "
                      "Identify the main types of work, key themes, and overall activity patterns."
        }]
        return self.chat(messages)

class CriticAgent(DeepSeekClient):
    """Report validation agent"""
    
    def __init__(self):
        super().__init__()
        self.set_system_prompt(
            "You are a strict quality control agent for Obsidian note summaries. "
            "Your task is to validate if the generated report follows all required rules. "
            "You will receive two inputs:\n"
            "1. The target LLM's system prompt.\n"
            "2. The response generated by the target LLM.\n"
            "You must verify whether the generated response adheres to the system prompt. "
            "If it does not, specify which aspect of the prompt was violated.\n"
            "Respond with a JSON object containing:\n"
            "{\n"
            '  "is_valid": boolean,\n'
            '  "issues": [list of issues found],\n'
            '  "suggestions": [list of improvement suggestions]\n'
            "}\n\n"
            "You should check for the following rules:\n"
            "1. Response must be in English\n"
            "2. Must include key information and concise summaries\n"
            "3. Must include these sections:\n"
            "   - Key keywords for each note\n"
            "   - Type of work performed (e.g., research, idea organization, meeting notes)\n"
            "   - Main content summary\n"
            "4. Each note's summary must be up to 4 sentences\n"
            "5. If the note is empty, must return 'Empty note'\n"
            "6. Must use bullet points for better readability\n"
            "7. Must be in markdown format"
        )

    def validate_report(self, report: str, target_prompt: str) -> Tuple[bool, List[str], List[str]]:
        """Validate report"""
        messages = [{
            "role": "user",
            "content": f"Please validate the following system prompt and generated report:\n\n"
                      f"System Prompt:\n{target_prompt}\n\n"
                      f"Generated Report:\n{report}"
        }]
        response = self.chat(messages)
        
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

def validate_report_with_retry(
    report: str,
    critic: CriticAgent,
    target_prompt: str,
    max_attempts: int = 3,
    on_validation_failed: Optional[callable] = None
) -> Tuple[bool, str, List[str], List[str]]:
    """
    Validate report and retry if necessary.
    
    Args:
        report: Report to validate
        critic: Validation agent
        target_prompt: System prompt of the target LLM
        max_attempts: Maximum number of retry attempts
        on_validation_failed: Callback function to call when validation fails (takes report, issues, suggestions as arguments)
    
    Returns:
        Tuple[bool, str, List[str], List[str]]: (success, final report, issues found, improvement suggestions)
    """
    for attempt in range(max_attempts):
        is_valid, issues, suggestions = critic.validate_report(report, target_prompt)
        
        if is_valid:
            return True, report, issues, suggestions
        
        print(f"\nAttempt {attempt + 1}/{max_attempts}")
        print("Issues found:")
        for issue in issues:
            print(f"- {issue}")
        print("\nImprovement suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
        
        if attempt < max_attempts - 1:
            print("\nRetrying with modified report...")
            if on_validation_failed:
                report = on_validation_failed(report, issues, suggestions)
        else:
            print("\nMaximum attempts reached. Returning the last report.")
    
    return False, report, issues, suggestions

def on_validation_failed(generator: GeneratorAgent, issues: List[str], suggestions: List[str]) -> str:
    """Modify report when validation fails"""
    return generator.chat([{
        "role": "user",
        "content": f"The previous report had the following issues. Please revise it:\n" + 
                    "\n".join(f"- {issue}" for issue in issues) + "\n\n" +
                    "Improvement suggestions:\n" +
                    "\n".join(f"- {suggestion}" for suggestion in suggestions)
    }])

def summarize_file_with_validation(
    generator: GeneratorAgent,
    critic: CriticAgent,
    summarize_func: callable,
    *args,
    max_attempts: int = 3
) -> str:
    """
    Generate and validate file summary.
    
    Args:
        generator: Summary generation agent
        critic: Validation agent
        summarize_func: Summary generation function (generator's method)
        *args: Arguments to pass to summarize_func
        max_attempts: Maximum number of retry attempts
    
    Returns:
        str: Validated summary
    """
    summary = summarize_func(*args)
    
    def on_summary_validation_failed(summary: str, issues: List[str], suggestions: List[str]) -> str:
        return generator.chat([{
            "role": "user",
            "content": f"The previous summary had the following issues. Please revise it:\n" + 
                      "\n".join(f"- {issue}" for issue in issues) + "\n\n" +
                      "Improvement suggestions:\n" +
                      "\n".join(f"- {suggestion}" for suggestion in suggestions)
        }])
    
    is_valid, final_summary, issues, suggestions = validate_report_with_retry(
        summary,
        critic,
        generator.system_prompt,
        max_attempts,
        lambda s, i, sug: on_summary_validation_failed(s, i, sug)
    )
    
    return final_summary

def summarize_daily_notes(
    created_files: List[DocumentData],
    modified_files: List[Tuple[DocumentData, List[str]]],
    deleted_files: List[str],  # List of file paths
    max_attempts: int = 3
) -> str:
    """Summarize daily notes (with validation)"""
    if not any([created_files, modified_files, deleted_files]):
        return "No notes were modified today."

    generator = GeneratorAgent()
    critic = CriticAgent()
    
    # 1. Generate summaries for each file (with validation)
    created_summaries = [
        (doc, summarize_file_with_validation(
            generator,
            critic,
            generator.summarize_file_content,
            doc,
            max_attempts=max_attempts
        )) for doc in created_files
    ]
    
    modified_summaries = [
        (doc, summarize_file_with_validation(
            generator,
            critic,
            generator.summarize_file_changes,
            doc,
            changes,
            max_attempts=max_attempts
        )) for doc, changes in modified_files
    ]
    
    deleted_summaries = [
        (path, summarize_file_with_validation(
            generator,
            critic,
            generator.summarize_deleted_file,
            path,
            max_attempts=max_attempts
        )) for path in deleted_files
    ]
    
    # 2. Generate and validate final daily report
    report = generator.generate_daily_report(
        created_summaries,
        modified_summaries,
        deleted_summaries
    )
    
    is_valid, final_report, issues, suggestions = validate_report_with_retry(
        report,
        critic,
        generator.system_prompt,
        max_attempts,
        lambda r, i, sug: on_validation_failed(generator, i, sug)
    )
    
    return final_report

if __name__ == "__main__":
    from pathlib import Path
    from parser import parse_markdown
    from watcher import get_recent_changes
    
    # Test execution
    vault_dir = Path.home() / "Documents" / "Obsidian Vault"
    created, deleted, modified = get_recent_changes(vault_dir)

    print("====================================")
    print("Created:")
    print(created)
    print("====================================")
    print("Deleted:")
    print(deleted)
    print("====================================")
    print("Modified:")
    print(modified)
    print("====================================")
    
    # Parse files
    created_docs = [parse_markdown(Path(p)) for p in created]
    modified_docs = [(parse_markdown(Path(p)), changes) for p, changes in modified.items()]
    
    print(summarize_daily_notes(created_docs, modified_docs, deleted))
