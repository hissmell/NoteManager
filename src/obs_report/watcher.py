# src/obs_report/watcher.py

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import difflib
import os

def get_state_file() -> Path:
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".obs_summary" / "state.json"

def load_previous_state() -> Dict[str, str]:
    state_file = get_state_file()
    if not state_file.exists():
        return {}
    return json.loads(state_file.read_text(encoding="utf-8"))

def save_state(state: Dict[str, str]):
    state_file = get_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def scan_vault(vault: Path) -> Dict[str, str]:
    """
    vault 내 모든 .md 파일을 읽어
    { str(path): content } dict 반환
    
    vault 내 모든 하위 폴더를 재귀적으로 탐색하며 .md 파일을 찾습니다.
    단, "Calendar" 폴더의 하위 폴더는 제외합니다.
    """
    files = {}
    for md in vault.rglob("*.md"):  # rglob()으로 모든 하위 폴더 재귀 탐색
        # Calendar 폴더의 하위 폴더인 경우 건너뛰기
        if "Calendar" in md.parts and md.parts.index("Calendar") < len(md.parts) - 1:
            continue
        files[str(md)] = md.read_text(encoding="utf-8")
    return files

def diff_contents(old: str, new: str) -> List[str]:
    """
    이전(old)과 현재(new) 내용을 line 단위로 비교해
    LLM이 이해하기 쉬운 형식의 변경사항 리스트 반환
    """
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    
    # difflib의 SequenceMatcher를 사용하여 변경사항 감지
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changes = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'delete':
            changes.append(f"삭제된 내용:\n" + "\n".join(f"  {line}" for line in old_lines[i1:i2]))
        elif tag == 'insert':
            changes.append(f"추가된 내용:\n" + "\n".join(f"  {line}" for line in new_lines[j1:j2]))
        elif tag == 'replace':
            changes.append(f"변경된 내용:\n" + 
                         f"  이전: {old_lines[i1:i2][0] if i2-i1 == 1 else '여러 줄'}\n" +
                         f"  이후: {new_lines[j1:j2][0] if j2-j1 == 1 else '여러 줄'}")
    
    return changes

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

def get_recent_changes(vault_path: Path, hours: int = 24, exclude_dirs: List[str] = None, update_state: bool = True):
    """
    Vault의 최근 변경사항을 감지합니다.
    
    Args:
        vault_path: Obsidian Vault 경로
        hours: 최근 몇 시간 동안의 변경사항을 볼지 (기본값: 24)
        exclude_dirs: 변경사항을 무시할 디렉토리 목록 (기본값: None)
        update_state: 현재 상태를 state.json에 저장할지 여부 (기본값: True)
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
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
    recent_created = {} # path: content
    recent_deleted = {} # path: content
    recent_modified = {} # path: content
    
    def should_exclude(path: str) -> bool:
        """파일이 제외할 디렉토리 하위에 있는지 확인"""
        path_parts = Path(path).parts
        return any(exclude_dir in path_parts for exclude_dir in exclude_dirs)
    
    for path in created:
        if should_exclude(path):
            continue
            
        file_time = datetime.fromtimestamp(Path(path).stat().st_mtime)
        print(f"{path} File time: {file_time}")
        if file_time > cutoff:
            recent_created[path] = curr_state[path]
    
    for path in deleted:
        if should_exclude(path):
            continue
        recent_deleted[path] = prev_state[path]
            
    for path, diff in modified.items():
        if should_exclude(path):
            continue
            
        file_time = datetime.fromtimestamp(Path(path).stat().st_mtime)
        if file_time > cutoff:
            recent_modified[path] = diff
    
    # 4. 결과 리턴
    print(f"Created (last {hours}h):", recent_created)
    print("Deleted:", recent_deleted)
    print(f"Modified (last {hours}h):")
    for path, diff in recent_modified.items():
        print(f"\n--- {path} ---")
        print("\n".join(diff))
        
    # 5. 스냅샷 갱신 (update_state가 True일 때만)
    if (not os.path.isfile(get_state_file())) or update_state:
        save_state(curr_state)

    return recent_created, recent_deleted, recent_modified

if __name__ == "__main__":
    vault_dir = Path.home() / "Documents" / "Obsidian Vault"
    ignore_dirs = os.path.join("Calendar", "Daily Report")
    recent_created, recent_deleted, recent_modified = get_recent_changes(vault_dir, exclude_dirs=[ignore_dirs])

    print("====================================")
    for path in recent_created:
        print(f"Created: {path}")

    print("====================================")
    for path in recent_deleted:
        print(f"Deleted: {path}")

    print("====================================")
    for path, diff in recent_modified.items():
        print(f"Modified: {path}")
        print("\n".join(diff))
