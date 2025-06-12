from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import json
import os

class LLMConfig(BaseModel):
    """LLM 설정"""
    model_name: str = Field("deepseek-r1:14b", description="LLM model name")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Temperature for text generation")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate (None for unlimited)")
    max_attempts: int = Field(3, ge=1, le=5, description="Maximum number of retry attempts")
    timeout: int = Field(120, ge=30, le=300, description="Timeout in seconds")

class AppConfig(BaseModel):
    """애플리케이션 설정"""
    vault_dir: Path = Field(..., description="Obsidian Vault 디렉토리 경로")
    report_dir: Path = Field(..., description="일간 보고서 저장 디렉토리 경로")
    llm_config: LLMConfig = Field(default_factory=LLMConfig, description="LLM 설정")
    exclude_dirs: List[str] = Field(default_factory=list, description="제외할 디렉토리 목록")
    cutoff_hours: int = Field(24, ge=1, le=168, description="변경사항 감지 기간 (시간)")

    @classmethod
    def get_config_path(cls) -> Path:
        """설정 파일 경로 반환"""
        return Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".obs_summary" / "config.json"

    def save(self) -> None:
        """설정을 파일에 저장"""
        config_path = self.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Path 객체를 문자열로 변환
        config_dict = self.model_dump()
        config_dict["vault_dir"] = str(config_dict["vault_dir"])
        config_dict["report_dir"] = str(config_dict["report_dir"])
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> Optional["AppConfig"]:
        """설정 파일에서 설정 로드"""
        config_path = cls.get_config_path()
        if not config_path.exists():
            return None
            
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        # 문자열 경로를 Path 객체로 변환
        config_dict["vault_dir"] = Path(config_dict["vault_dir"])
        config_dict["report_dir"] = Path(config_dict["report_dir"])
        
        return cls(**config_dict) 