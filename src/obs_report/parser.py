# src/obs_report/parser.py

from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
import frontmatter
import markdown
from bs4 import BeautifulSoup

class DocumentData(BaseModel):
    path: str
    metadata: Dict[str, Any]
    title: str
    headers: List[str]
    content: str

def parse_markdown(file_path: Path) -> DocumentData:
    """
    파일 경로의 Markdown 문서를 읽어서
    - frontmatter.metadata
    - title (첫 번째 '# ' 헤더)
    - headers (모든 헤더 텍스트)
    - content (본문 전체, frontmatter 제외)
    
    를 포함하는 DocumentData 객체로 반환합니다.
    """
    # 1) frontmatter 파싱
    post = frontmatter.load(file_path.read_text(encoding="utf-8"))
    meta = post.metadata

    # 2) 본문(마크다운) → HTML → BeautifulSoup로 파싱
    html = markdown.markdown(post.content)
    soup = BeautifulSoup(html, "html.parser")

    # 3) 제목 & 모든 헤더 수집
    headers = [h.get_text().strip() for h in soup.find_all(['h1','h2','h3','h4','h5','h6'])]
    title = headers[0] if headers else file_path.stem

    # 4) 순수 텍스트 컨텐츠 (HTML 태그 제거)
    body_text = soup.get_text(separator="\n").strip()

    return DocumentData(
        path=str(file_path),
        metadata=meta,
        title=title,
        headers=headers,
        content=body_text
    )

if __name__ == "__main__":
    from pathlib import Path
    # 테스트용 실행
    test_file = Path.home() / "Documents/Obsidian Vault/Projects/Personal/NoteManager/Road map.md"
    doc = parse_markdown(test_file)
    print(doc.json(indent=2, ensure_ascii=False))
