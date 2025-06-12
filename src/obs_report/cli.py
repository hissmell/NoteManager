import typer
from pathlib import Path
from typing import Optional
from .config import AppConfig, LLMConfig

app = typer.Typer(help="Obsidian 노트 관리 자동화 도구")

@app.command()
def init(
    vault_dir: Path = typer.Option(
        ...,
        "--vault-dir",
        "-v",
        help="Obsidian Vault 디렉토리 경로",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    report_dir: Path = typer.Option(
        ...,
        "--report-dir",
        "-r",
        help="일간 보고서 저장 디렉토리 경로",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    model_name: str = typer.Option(
        "deepseek-r1:14b",
        "--model",
        "-m",
        help="사용할 LLM 모델 이름",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help="LLM temperature 값 (0.0 ~ 1.0)",
        min=0.0,
        max=1.0,
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        help="최대 토큰 수",
        min=1,
    ),
    max_attempts: int = typer.Option(
        3,
        "--max-attempts",
        help="최대 재시도 횟수",
        min=1,
    ),
    timeout: int = typer.Option(
        120,
        "--timeout",
        help="요청 타임아웃 (초)",
        min=1,
    ),
    exclude_dirs: Optional[list[str]] = typer.Option(
        None,
        "--exclude-dir",
        "-e",
        help="제외할 디렉토리 목록",
    ),
):
    """초기 환경 설정을 진행합니다."""
    # LLM 설정 생성
    llm_config = LLMConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_attempts=max_attempts,
        timeout=timeout,
    )
    
    # 앱 설정 생성
    app_config = AppConfig(
        vault_dir=vault_dir,
        report_dir=report_dir,
        llm=llm_config,
        exclude_dirs=exclude_dirs or [],
    )
    
    # 설정 저장
    app_config.save()
    
    typer.echo("설정이 저장되었습니다:")
    typer.echo(f"Vault 디렉토리: {vault_dir}")
    typer.echo(f"보고서 디렉토리: {report_dir}")
    typer.echo(f"LLM 모델: {model_name}")
    typer.echo(f"Temperature: {temperature}")
    typer.echo(f"최대 토큰 수: {max_tokens}")
    typer.echo(f"최대 재시도 횟수: {max_attempts}")
    typer.echo(f"타임아웃: {timeout}초")
    if exclude_dirs:
        typer.echo(f"제외 디렉토리: {', '.join(exclude_dirs)}")

@app.command()
def show():
    """현재 설정을 보여줍니다."""
    config = AppConfig.load()
    if config is None:
        typer.echo("설정이 없습니다. 'init' 명령어로 초기 설정을 진행해주세요.")
        raise typer.Exit(1)
        
    typer.echo("현재 설정:")
    typer.echo(f"Vault 디렉토리: {config.vault_dir}")
    typer.echo(f"보고서 디렉토리: {config.report_dir}")
    typer.echo(f"LLM 모델: {config.llm.model_name}")
    typer.echo(f"Temperature: {config.llm.temperature}")
    typer.echo(f"최대 토큰 수: {config.llm.max_tokens}")
    typer.echo(f"최대 재시도 횟수: {config.llm.max_attempts}")
    typer.echo(f"타임아웃: {config.llm.timeout}초")
    if config.exclude_dirs:
        typer.echo(f"제외 디렉토리: {', '.join(config.exclude_dirs)}")

if __name__ == "__main__":
    app() 