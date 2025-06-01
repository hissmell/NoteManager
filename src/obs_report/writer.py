# src/obs_report/writer.py

from pathlib import Path
from datetime import datetime, date, timedelta
from llm_clinet import summarize_daily_notes

def write_daily_note(summary: str, vault_path: Path) -> Path:
    """
    Vault/Calendar/01_Daily/YYYY-MM-DD.md 파일에 '## 오늘 요약' 섹션을 최신화하거나 생성합니다.
    Returns: 실제로 기록된 파일 경로
    """
    today = date.today().isoformat()  # 예: "2025-06-05"
    daily_dir = vault_path / "Calendar" / "01_Daily"
    daily_dir.mkdir(exist_ok=True)

    note_file = daily_dir / f"{today}.md"
    _write_section(note_file, section_title="## 오늘 요약", content=summary)
    return note_file

def write_weekly_report(summary: str, vault_path: Path) -> Path:
    """
    Vault/Calendar/02_Weekly/YYYY-WW.md 파일(ISO 주차 사용)에 '## 주간 요약' 섹션을 최신화하거나 생성합니다.
    Returns: 실제로 기록된 파일 경로
    """
    # ISO 주차 구하기 (ex: "2025-W23")
    today = date.today()
    iso_year, iso_week, _ = today.isocalendar()
    week_str = f"{iso_year}-W{iso_week:02d}"
    weekly_dir = vault_path / "Calendar" / "02_Weekly"
    weekly_dir.mkdir(exist_ok=True)

    report_file = weekly_dir / f"{week_str}.md"
    _write_section(report_file, section_title="## 주간 요약", content=summary)
    return report_file

def write_monthly_report(summary: str, vault_path: Path) -> Path:
    """
    Vault/Calendar/03_Monthly/YYYY-MM.md 파일에 '## 월간 요약' 섹션을 최신화하거나 생성합니다.
    Returns: 실제로 기록된 파일 경로
    """
    today = date.today()
    month_str = today.strftime("%Y-%m")  # 예: "2025-06"
    monthly_dir = vault_path / "Calendar" / "03_Monthly"
    monthly_dir.mkdir(exist_ok=True)

    report_file = monthly_dir / f"{month_str}.md"
    _write_section(report_file, section_title="## 월간 요약", content=summary)
    return report_file

def _write_section(file_path: Path, section_title: str, content: str):
    """
    - file_path가 없으면 디렉터리까지 생성 후 새 파일 생성
    - 있으면, 기존 파일을 읽어서 해당 section_title 아래 내용을 교체
    """
    # 1) 파일이 없으면 빈 틀 생성
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")

    # 2) 기존 파일 읽기
    lines = file_path.read_text(encoding="utf-8").splitlines()

    # 3) section_title 위치 찾기 (있으면 인덱스, 없으면 -1)
    try:
        idx = next(i for i, line in enumerate(lines) if line.strip() == section_title)
    except StopIteration:
        idx = -1

    if idx == -1:
        # section_title이 없으면 파일 맨 아래에 새 섹션 추가
        new_lines = lines + ["", section_title, "", content.strip(), ""]
    else:
        # 기존 섹션을 지우고 새로 교체
        # section 이후부터 다음 '## ' (같은 레벨 헤더) 전까지를 제거
        # 같은 레벨 헤더가 하나 더 있는 경우: 그 인덱스 찾기
        end_idx = len(lines)
        for j in range(idx + 1, len(lines)):
            if lines[j].startswith("## ") and j != idx:
                end_idx = j
                break
        # idx부터 end_idx-1까지 버리고, section_title → content 삽입
        new_lines = lines[:idx] + [section_title, "", content.strip(), ""] + lines[end_idx:]

    # 4) 덮어쓰기
    file_path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    from pathlib import Path
    from parser import parse_markdown
    from watcher import get_recent_changes
    
    # 테스트용 실행
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
    
    # 파일 파싱
    created_docs = [parse_markdown(Path(p)) for p in created]
    modified_docs = [(parse_markdown(Path(p)), changes) for p, changes in modified.items()]
    
    summary_str = summarize_daily_notes(created_docs, modified_docs, deleted)
    write_daily_note(summary_str, vault_dir)

