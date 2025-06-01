# tests/test_watcher.py

import json, os, sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
OBS_REPORT_DIR = os.path.join(PROJECT_DIR, "obs_report")

sys.path.append(PROJECT_DIR)
sys.path.append(SRC_DIR)
sys.path.append(OBS_REPORT_DIR)

import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pytest
from obs_report.watcher import save_state, load_previous_state, detect_changes, scan_vault

def test_save_and_load_state(tmp_path, monkeypatch):
    # 1) 임시 디렉터리를 HOME 으로 설정
    fake_home = tmp_path / "home"
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    # 2) 테스트용 데이터
    data = {
        "a.md": "# Note A",
        "subdir/b.md": "Content B"
    }
    # 3) save_state 호출
    save_state(data)
    # 4) 파일이 생성됐는지, 내용이 JSON 으로 맞게 들어갔는지 확인
    state_file = fake_home / ".obs_summary" / "state.json"
    assert state_file.exists()
    loaded = load_previous_state()
    # 5) 읽어 온 데이터가 원본과 동일해야 함
    assert loaded == data

def test_detect_changes_simple():
    prev = {
        "a.md": "line1\nline2\n",
        "b.md": "foo\nbar\n"
    }
    curr = {
        "b.md": "foo\nbaz\n",    # 수정된 라인2
        "c.md": "new file\n"     # 새로 생성
    }

    created, deleted, modified = detect_changes(prev, curr)

    # created/deleted
    assert created == ["c.md"]
    assert deleted == ["a.md"]

    # modified 에 'b.md' 가 있고,
    # diff 에서 기존 'bar' 라인이 '-' 로, 새로운 'baz' 라인이 '+' 로 표기되어야 함
    assert "b.md" in modified
    diff = modified["b.md"]
    assert any(line.startswith("-bar") for line in diff)
    assert any(line.startswith("+baz") for line in diff)

def test_scan_vault(tmp_path):
    # 1) tmp_path 에 간단한 폴더 구조와 .md 파일 생성
    vault = tmp_path / "vault"
    (vault / "sub").mkdir(parents=True)
    f1 = vault / "one.md"
    f2 = vault / "sub" / "two.md"
    f1.write_text("Content One", encoding="utf-8")
    f2.write_text("Content Two", encoding="utf-8")

    # 2) scan_vault 호출
    result = scan_vault(vault)

    # 3) 키(절대경로) 2개, 값(각 파일 텍스트) 2개가 정확히 들어 있어야 함
    expected_paths = {str(f1), str(f2)}
    assert set(result.keys()) == expected_paths
    assert result[str(f1)] == "Content One"
    assert result[str(f2)] == "Content Two"

def test_get_recent_changes(tmp_path, monkeypatch):
    # 1) 임시 디렉터리를 HOME 으로 설정
    fake_home = tmp_path / "home"
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    
    # 2) 테스트용 vault 생성
    vault = tmp_path / "vault"
    vault.mkdir()
    
    # 3) 테스트용 파일 생성
    old_file = vault / "old.md"
    new_file = vault / "new.md"
    modified_file = vault / "modified.md"
    
    old_file.write_text("Old content", encoding="utf-8")
    new_file.write_text("New content", encoding="utf-8")
    modified_file.write_text("Original content", encoding="utf-8")
    
    # 4) 이전 상태 저장
    initial_state = {
        str(old_file): "Old content",
        str(modified_file): "Original content"
    }
    save_state(initial_state)
    
    # 5) 파일 수정
    modified_file.write_text("Modified content", encoding="utf-8")
    
    # 6) get_recent_changes 호출
    from obs_report.watcher import get_recent_changes
    recent_created, recent_deleted, recent_modified = get_recent_changes(vault, hours=24)
    
    # 7) 결과 검증
    assert str(new_file) in recent_created
    assert str(modified_file) in recent_modified
    assert len(recent_deleted) == 0  # 삭제된 파일 없음