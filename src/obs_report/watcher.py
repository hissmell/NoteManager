# src/obs_report/watcher.py

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import difflib

STATE_FILE = Path.home() / ".obs_summary" / "state.json"

def load_previous_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))

def save_state(state: Dict[str, str]):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def scan_vault(vault: Path) -> Dict[str, str]:
    """
    vault 내 모든 .md 파일을 읽어
    { str(path): content } dict 반환
    
    vault 내 모든 하위 폴더를 재귀적으로 탐색하며 .md 파일을 찾습니다.
    """
    files = {}
    for md in vault.rglob("*.md"):  # rglob()으로 모든 하위 폴더 재귀 탐색
        files[str(md)] = md.read_text(encoding="utf-8")
    return files

def diff_contents(old: str, new: str) -> List[str]:
    """
    이전(old)과 현재(new) 내용을 line 단위로 비교해
    unified diff 형태의 리스트 반환
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    return list(difflib.unified_diff(old_lines, new_lines, lineterm=""))

def detect_changes(
    prev: Dict[str, str],
    curr: Dict[str, str]
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    created = [p for p in curr if p not in prev]
    deleted = [p for p in prev if p not in curr]
    modified = {}
    for p in curr:
        if p in prev and prev[p] != curr[p]:
            modified[p] = diff_contents(prev[p], curr[p])
    return created, deleted, modified

def get_recent_changes(vault_path: Path, hours: int = 24):
    # 1. 이전 상태 로드
    prev_state = load_previous_state()
    # 2. 현재 Vault 스캔
    curr_state = scan_vault(vault_path)
    # 3. 변경 감지
    created, deleted, modified = detect_changes(prev_state, curr_state)


    
    # 시간 필터링을 위한 기준 시간
    cutoff = datetime.now() - timedelta(hours=hours)

    print(f"Cutoff: {cutoff}")
    
    # 최근 생성/수정된 파일만 필터링
    recent_created = []
    recent_deleted = []
    recent_modified = {}
    
    for path in created:
        file_time = datetime.fromtimestamp(Path(path).stat().st_mtime)

        print(f"{path} File time: {file_time}")
        if file_time > cutoff:
            recent_created.append(path)
    
    for path in deleted:
        recent_deleted.append(path)
            
    for path, diff in modified.items():
        file_time = datetime.fromtimestamp(Path(path).stat().st_mtime)
        if file_time > cutoff:
            recent_modified[path] = diff
    
    # 4. 결과 리턴
    print(f"Created (last {hours}h):", recent_created)
    print("Deleted:", deleted)
    print(f"Modified (last {hours}h):")
    for path, diff in recent_modified.items():
        print(f"\n--- {path} ---")
        print("".join(diff))
        
    # 5. 스냅샷 갱신
    save_state(curr_state)

    return recent_created, recent_deleted, recent_modified

if __name__ == "__main__":
    vault_dir = Path.home() / "Documents" / "Obsidian Vault"
    recent_created, recent_deleted, recent_modified = get_recent_changes(vault_dir)

    print("====================================")
    for path in recent_created:
        print(f"Created: {path}")

    print("====================================")
    for path in recent_deleted:
        print(f"Deleted: {path}")

    print("====================================")
    for path, diff in recent_modified.items():
        print(f"Modified: {path}")
        print("".join(diff))
