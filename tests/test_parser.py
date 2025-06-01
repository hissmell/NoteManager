# tests/test_parser.py
import json, os, sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
OBS_REPORT_DIR = os.path.join(PROJECT_DIR, "obs_report")

sys.path.append(PROJECT_DIR)
sys.path.append(SRC_DIR)
sys.path.append(OBS_REPORT_DIR)

import pytest
from pathlib import Path
from obs_report.parser import parse_markdown, DocumentData

# 1. Frontmatter 포함 + 헤더/본문 있는 경우
SAMPLE_WITH_FM = """---
title: Test Note
tags:
  - python
  - obsidian
---

# Heading One

Some **bold** text.

## Subheading

More text here.
"""

# 2. Frontmatter 없이 단순 헤더 + 본문만 있는 경우
SAMPLE_NO_FM = """
# Only Heading

Just some text without frontmatter.
"""

# 3. 헤더가 전혀 없는 경우 (본문만)
SAMPLE_NO_HEADER = """
This is a note without any Markdown headers.
Just plain text.
"""

@pytest.mark.parametrize("content, expected_meta, expected_title, expected_headers, expected_snippet", [
    # case 1: frontmatter + 여러 헤더
    (
        SAMPLE_WITH_FM,
        {"title": "Test Note", "tags": ["python", "obsidian"]},
        "Heading One",
        ["Heading One", "Subheading"],
        "Some bold text."  # 본문 일부
    ),
    # case 2: frontmatter 없이 하나의 헤더만
    (
        SAMPLE_NO_FM,
        {},                # Frontmatter가 없으면 metadata는 빈 dict
        "Only Heading",
        ["Only Heading"],
        "Just some text without frontmatter."
    ),
    # case 3: 헤더가 전혀 없는 경우
    (
        SAMPLE_NO_HEADER,
        {},
        "test_parser",     # 파일명이 test_parser.md 라 가정 (경로 .stem 사용)
        [],                # headers 빈 리스트
        "This is a note without any Markdown headers."
    ),
])
def test_parse_markdown_various(tmp_path, content, expected_meta, expected_title, expected_headers, expected_snippet):
    # 1) 임시 .md 파일 생성
    test_file = tmp_path / "test_parser.md"
    test_file.write_text(content, encoding="utf-8")

    # 2) parse_markdown 호출
    doc: DocumentData = parse_markdown(test_file)

    # 3) 반환된 DocumentData 속성 검증
    assert isinstance(doc, DocumentData)
    #   3.1) path
    assert doc.path.endswith("test_parser.md")
    #   3.2) metadata
    assert doc.metadata == expected_meta
    #   3.3) title
    assert doc.title == expected_title
    #   3.4) headers
    assert doc.headers == expected_headers
    #   3.5) content에 일부 키워드 포함
    assert expected_snippet in doc.content
