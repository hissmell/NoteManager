from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QListWidget, QGroupBox, QComboBox,
    QCheckBox, QProgressDialog, QListWidgetItem, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from pathlib import Path
from typing import Optional, List, Dict
from obs_report.config import AppConfig, LLMConfig
from obs_report.llm_clinet import BaseLLMClient, DeepSeekClient, Phi4Client, GeneratorAgent, CriticAgent, NoteManager
from obs_report.watcher import get_recent_changes
from obs_report.parser import parse_markdown

class SummaryWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, vault_dir: Path, report_dir: Path, llm_config: LLMConfig, exclude_dirs: List[str], selected_notes: Dict[str, List[str]], cutoff_hours: int):
        super().__init__()
        self.vault_dir = vault_dir
        self.report_dir = report_dir
        self.llm_config = llm_config
        self.exclude_dirs = exclude_dirs
        self.selected_notes = selected_notes
        self.cutoff_hours = cutoff_hours
        
        # 모델에 따라 적절한 클라이언트 생성
        if llm_config.model_name == "deepseek-r1:14b":
            self.llm_client = DeepSeekClient()
        elif llm_config.model_name == "phi4:latest":
            self.llm_client = Phi4Client()
        else:
            raise ValueError(f"지원하지 않는 모델: {llm_config.model_name}")
        
        # LLM 설정 적용
        self.llm_client.temperature = llm_config.temperature
        self.llm_client.max_tokens = llm_config.max_tokens
        self.llm_client.timeout = llm_config.timeout
        self.llm_client.max_retries = llm_config.max_attempts
        
        # GeneratorAgent와 CriticAgent 생성
        self.generator = GeneratorAgent(self.llm_client.clone())
        self.critic = CriticAgent(self.llm_client.clone())
        
        # NoteManager 생성
        self.manager = NoteManager(self.generator, self.critic)
    
    def run(self):
        try:
            # 선택된 노트만 처리
            created = self.selected_notes.get("created", {})
            modified = self.selected_notes.get("modified", {})
            deleted = self.selected_notes.get("deleted", {})

            for i, d in enumerate(modified):
                print(f"modified {i}: {d}")
            
            # 파일 파싱
            self.progress.emit("파일 파싱 중...")
            try:
                created_docs = [parse_markdown(Path(p), content=d["content"], changes=d["changes"]) for p,d in created.items()]
                modified_docs = [parse_markdown(Path(p), content=d["content"], changes=d["changes"]) for p,d in modified.items()]
                deleted_docs = [parse_markdown(Path(p), content=d["content"], changes=d["changes"]) for p,d in deleted.items()]
            except Exception as e:
                raise Exception(f"파일 파싱 중 오류 발생: {str(e)}")
            
            # 생성된 파일 처리
            if created_docs:
                self.progress.emit(f"새로 생성된 노트 {len(created_docs)}개 처리 중...")
                for i, doc in enumerate(created_docs, 1):
                    try:
                        self.progress.emit(f"새로 생성된 노트 처리 중 ({i}/{len(created_docs)}): {Path(doc.path).name}")
                        self.manager._summarize_file(doc)
                    except Exception as e:
                        raise Exception(f"새로 생성된 노트 '{Path(doc.path).name}' 처리 중 오류 발생: {str(e)}")
            
            # 수정된 파일 처리
            if modified_docs:
                self.progress.emit(f"수정된 노트 {len(modified_docs)}개 처리 중...")
                for i, doc in enumerate(modified_docs, 1):
                    try:
                        self.progress.emit(f"수정된 노트 처리 중 ({i}/{len(modified_docs)}): {Path(doc.path).name}")
                        self.manager._summarize_file_changes(doc)
                    except Exception as e:
                        raise Exception(f"수정된 노트 '{Path(doc.path).name}' 처리 중 오류 발생: {str(e)}")
            
            # 삭제된 파일 처리
            if deleted_docs:
                self.progress.emit(f"삭제된 노트 {len(deleted_docs)}개 처리 중...")
                for i, doc in enumerate(deleted_docs, 1):
                    try:
                        self.progress.emit(f"삭제된 노트 처리 중 ({i}/{len(deleted_docs)}): {Path(doc.path).name}")
                        self.manager._summarize_deleted_file(doc)
                    except Exception as e:
                        raise Exception(f"삭제된 노트 '{Path(doc.path).name}' 처리 중 오류 발생: {str(e)}")
            
            # 일간 보고서 생성
            try:
                self.progress.emit("일간 보고서 생성 중...")
                self.manager.write_daily_report(
                    self.report_dir,
                    created_docs,
                    modified_docs,
                    deleted_docs
                )
            except Exception as e:
                raise Exception(f"일간 보고서 생성 중 오류 발생: {str(e)}")
            
            self.finished.emit()
            
        except Exception as e:
            import traceback
            error_msg = f"오류 발생:\n{str(e)}\n\n상세 정보:\n{traceback.format_exc()}"
            self.error.emit(error_msg)

def show_config_window():
    """설정 창을 표시합니다."""
    window = ConfigWindow()
    window.show()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Obsidian Report Generator")
        self.setMinimumSize(800, 600)
        
        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        layout = QVBoxLayout(central_widget)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 설정 버튼
        settings_button = QPushButton("설정")
        settings_button.clicked.connect(self.show_settings)
        button_layout.addWidget(settings_button)
        
        # 상태 업데이트 버튼
        update_state_button = QPushButton("update_state.json")
        update_state_button.clicked.connect(self.update_state)
        button_layout.addWidget(update_state_button)
        
        layout.addLayout(button_layout)
        
        # 노트 목록 탭
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 생성된 노트 탭
        self.created_tab = QWidget()
        self.created_layout = QVBoxLayout(self.created_tab)
        self.created_list = QListWidget()
        self.created_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.created_layout.addWidget(self.created_list)
        self.tab_widget.addTab(self.created_tab, "생성된 노트")
        
        # 수정된 노트 탭
        self.modified_tab = QWidget()
        self.modified_layout = QVBoxLayout(self.modified_tab)
        self.modified_list = QListWidget()
        self.modified_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.modified_layout.addWidget(self.modified_list)
        self.tab_widget.addTab(self.modified_tab, "수정된 노트")
        
        # 삭제된 노트 탭
        self.deleted_tab = QWidget()
        self.deleted_layout = QVBoxLayout(self.deleted_tab)
        self.deleted_list = QListWidget()
        self.deleted_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.deleted_layout.addWidget(self.deleted_list)
        self.tab_widget.addTab(self.deleted_tab, "삭제된 노트")
        
        # 요약 버튼
        summary_button = QPushButton("요약 실행")
        summary_button.clicked.connect(self.run_summary)
        layout.addWidget(summary_button)
        
        # 설정 창 참조
        self.config_window = None
        
        # 설정 로드
        self.load_config()
        
        # 변경 감지 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_changes)
        self.timer.start(60000)  # 1분마다 체크
    
    def show_settings(self):
        """설정 창을 표시합니다."""
        if self.config_window is None:
            self.config_window = ConfigWindow()
        self.config_window.show()
        self.load_config()
    
    def load_config(self):
        """설정을 로드합니다."""
        self.config = AppConfig.load()
        if not self.config:
            # 설정이 없을 때는 설정 창을 표시
            if self.config_window is None:
                self.config_window = ConfigWindow()
            self.config_window.show()
        else:
            self.detect_changes()
    
    def detect_changes(self):
        """변경사항 감지"""
        if not self.config:
            return
        
        # 변경사항 감지
        created, deleted, modified = get_recent_changes(
            self.config.vault_dir,
            exclude_dirs=self.config.exclude_dirs,
            cutoff_hours=self.config.cutoff_hours,
            update_state=False
        )
        # 리스트 업데이트
        self.created_list.clear()
        for note in created:
            item = QListWidgetItem(note)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.created_list.addItem(item)
        
        self.modified_list.clear()
        for note in modified:
            item = QListWidgetItem(note)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.modified_list.addItem(item)
        
        self.deleted_list.clear()
        for note in deleted:
            item = QListWidgetItem(note)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.deleted_list.addItem(item)
    
    def get_selected_notes(self):
        """선택된 노트 목록 반환"""
        # 변경사항 감지
        created, deleted, modified = get_recent_changes(
            self.config.vault_dir,
            exclude_dirs=self.config.exclude_dirs,
            cutoff_hours=self.config.cutoff_hours,
            update_state=False
        )

        for i, d in enumerate(modified):
            print(f"h1h1 modified {i}: {d}")
        
        # 선택된 노트만 필터링
        selected_notes = {
            "created": {},
            "deleted": {},
            "modified": {}
        }
        
        # 생성된 노트
        for i in range(self.created_list.count()):
            item = self.created_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                note_path = item.text()
                if note_path in created:
                    selected_notes["created"][note_path] = created[note_path]
        
        # 삭제된 노트
        for i in range(self.deleted_list.count()):
            item = self.deleted_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                note_path = item.text()
                if note_path in deleted:
                    selected_notes["deleted"][note_path] = deleted[note_path]
        
        # 수정된 노트
        for i in range(self.modified_list.count()):
            item = self.modified_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                note_path = item.text()
                if note_path in modified:
                    selected_notes["modified"][note_path] = modified[note_path]
        
        return selected_notes
    
    def run_summary(self):
        if not self.config:
            QMessageBox.warning(self, "경고", "설정이 필요합니다.")
            self.show_settings()
            return
        
        # 제외할 노트 확인
        excluded_notes = self.get_selected_notes()
        print(f"excluded_notes: {excluded_notes}")
        total_excluded = sum(len(notes) for notes in excluded_notes.values())
        
        if total_excluded == 0:
            # 모든 노트를 포함
            created, deleted, modified = get_recent_changes(
                self.config.vault_dir,
                exclude_dirs=self.config.exclude_dirs,
                cutoff_hours=self.config.cutoff_hours,
                update_state=False
            )
            excluded_notes = {
                "created": created,
                "deleted": deleted,
                "modified": modified
            }
        
        # 작업자 스레드 시작
        self.worker = SummaryWorker(
            self.config.vault_dir,
            self.config.report_dir,
            self.config.llm_config,
            self.config.exclude_dirs,
            excluded_notes,
            self.config.cutoff_hours
        )
        
        # 진행 상황 다이얼로그
        self.progress = QProgressDialog("요약 중...", "취소", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.setAutoReset(True)
        self.progress.setMinimumDuration(0)  # 즉시 표시
        
        # 시그널 연결
        self.worker.progress.connect(self.progress.setLabelText)
        self.worker.finished.connect(self.on_summary_finished)
        self.worker.error.connect(self.on_summary_error)
        
        # 작업 시작
        self.worker.start()
        self.progress.show()
    
    def on_summary_finished(self):
        if self.progress:
            self.progress.close()
        QMessageBox.information(self, "완료", "노트 요약이 완료되었습니다.")
    
    def on_summary_error(self, error_msg):
        if self.progress:
            self.progress.close()
        QMessageBox.critical(self, "오류", f"요약 중 오류가 발생했습니다:\n{error_msg}")

    def update_state(self):
        """state.json 파일을 현재 상태로 업데이트"""
        if not self.config:
            QMessageBox.warning(self, "경고", "설정이 필요합니다.")
            self.show_settings()
            return
        
        try:
            # 변경사항 감지 및 state.json 업데이트
            get_recent_changes(
                self.config.vault_dir,
                exclude_dirs=self.config.exclude_dirs,
                cutoff_hours=self.config.cutoff_hours,
                update_state=True
            )
            QMessageBox.information(self, "완료", "state.json이 업데이트되었습니다.")
            
            # 목록 새로고침
            self.detect_changes()
        except Exception as e:
            QMessageBox.critical(self, "오류", f"state.json 업데이트 중 오류가 발생했습니다:\n{str(e)}")

def show_main_window():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    show_main_window()

class ConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("설정")
        self.setMinimumWidth(400)
        
        # 메인 레이아웃
        layout = QVBoxLayout(self)
        
        # Vault 디렉토리 설정
        vault_group = QGroupBox("Obsidian Vault")
        vault_layout = QHBoxLayout()
        self.vault_edit = QLineEdit()
        self.vault_edit.setReadOnly(True)
        vault_button = QPushButton("찾아보기")
        vault_button.clicked.connect(self.select_vault_dir)
        vault_layout.addWidget(self.vault_edit)
        vault_layout.addWidget(vault_button)
        vault_group.setLayout(vault_layout)
        layout.addWidget(vault_group)
        
        # 보고서 디렉토리 설정
        report_group = QGroupBox("보고서 저장 디렉토리")
        report_layout = QHBoxLayout()
        self.report_edit = QLineEdit()
        self.report_edit.setReadOnly(True)
        report_button = QPushButton("찾아보기")
        report_button.clicked.connect(self.select_report_dir)
        report_layout.addWidget(self.report_edit)
        report_layout.addWidget(report_button)
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        # 제외할 디렉토리 설정
        exclude_group = QGroupBox("제외할 디렉토리")
        exclude_layout = QVBoxLayout()
        self.exclude_list = QListWidget()
        exclude_layout.addWidget(self.exclude_list)
        
        # 제외할 디렉토리 추가/삭제 버튼
        exclude_buttons = QHBoxLayout()
        add_exclude_button = QPushButton("추가")
        add_exclude_button.clicked.connect(self.add_exclude_dir)
        remove_exclude_button = QPushButton("삭제")
        remove_exclude_button.clicked.connect(self.remove_exclude_dir)
        exclude_buttons.addWidget(add_exclude_button)
        exclude_buttons.addWidget(remove_exclude_button)
        exclude_layout.addLayout(exclude_buttons)
        
        exclude_group.setLayout(exclude_layout)
        layout.addWidget(exclude_group)
        
        # LLM 설정
        llm_group = QGroupBox("LLM 설정")
        llm_layout = QVBoxLayout()
        
        # 모델 선택
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-r1:14b", "phi4:latest"])
        model_layout.addWidget(self.model_combo)
        llm_layout.addLayout(model_layout)
        
        # Temperature 설정
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.1)
        temp_layout.addWidget(self.temp_spin)
        llm_layout.addLayout(temp_layout)
        
        # Max Tokens 설정
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Tokens:"))
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(1, 4096)
        self.tokens_spin.setValue(1024)
        tokens_layout.addWidget(self.tokens_spin)
        llm_layout.addLayout(tokens_layout)
        
        # Timeout 설정
        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(QLabel("Timeout (초):"))
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 300)
        self.timeout_spin.setValue(120)
        timeout_layout.addWidget(self.timeout_spin)
        llm_layout.addLayout(timeout_layout)
        
        # Max Attempts 설정
        attempts_layout = QHBoxLayout()
        attempts_layout.addWidget(QLabel("Max Attempts:"))
        self.attempts_spin = QSpinBox()
        self.attempts_spin.setRange(1, 10)
        self.attempts_spin.setValue(3)
        attempts_layout.addWidget(self.attempts_spin)
        llm_layout.addLayout(attempts_layout)
        
        # Cutoff Hours 설정
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("변경사항 감지 기간 (시간):"))
        self.cutoff_spin = QSpinBox()
        self.cutoff_spin.setRange(1, 168)  # 1시간 ~ 1주일
        self.cutoff_spin.setValue(24)
        cutoff_layout.addWidget(self.cutoff_spin)
        llm_layout.addLayout(cutoff_layout)
        
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
        
        # 저장 버튼
        save_button = QPushButton("저장")
        save_button.clicked.connect(self.close)  # 저장 버튼 클릭 시 창 닫기
        layout.addWidget(save_button)
        
        # 설정 로드
        self.load_config()
    
    def select_vault_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Obsidian 볼트 디렉토리 선택")
        if dir_path:
            self.vault_edit.setText(dir_path)
    
    def select_report_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "보고서 저장 디렉토리 선택")
        if dir_path:
            self.report_edit.setText(dir_path)
    
    def load_config(self):
        config = AppConfig.load()
        if config:
            self.vault_edit.setText(str(config.vault_dir))
            self.report_edit.setText(str(config.report_dir))
            self.exclude_list.clear()
            self.exclude_list.addItems(config.exclude_dirs)
            self.model_combo.setCurrentText(config.llm_config.model_name)
            self.temp_spin.setValue(config.llm_config.temperature)
            self.tokens_spin.setValue(config.llm_config.max_tokens)
            self.timeout_spin.setValue(config.llm_config.timeout)
            self.attempts_spin.setValue(config.llm_config.max_attempts)
            self.cutoff_spin.setValue(config.cutoff_hours)
    
    def closeEvent(self, event):
        """창이 닫힐 때 호출되는 이벤트"""
        # 설정 저장
        vault_dir = Path(self.vault_edit.text())
        report_dir = Path(self.report_edit.text())
        
        if not vault_dir.exists():
            QMessageBox.warning(self, "경고", "Vault 디렉토리가 존재하지 않습니다.")
            event.ignore()
            return
        
        if not report_dir.exists():
            try:
                report_dir.mkdir(parents=True)
            except Exception as e:
                QMessageBox.warning(self, "경고", f"보고서 디렉토리 생성 실패: {str(e)}")
                event.ignore()
                return
        
        # 설정 저장
        config = AppConfig(
            vault_dir=vault_dir,
            report_dir=report_dir,
            exclude_dirs=[self.exclude_list.item(i).text() for i in range(self.exclude_list.count())],
            llm_config=LLMConfig(
                model_name=self.model_combo.currentText(),
                temperature=self.temp_spin.value(),
                max_tokens=self.tokens_spin.value(),
                timeout=self.timeout_spin.value(),
                max_attempts=self.attempts_spin.value()
            ),
            cutoff_hours=self.cutoff_spin.value()
        )
        config.save()
        QMessageBox.information(self, "알림", "설정이 저장되었습니다.")
        event.accept()
    
    def save_config(self):
        """설정 저장 (closeEvent에서 호출됨)"""
        self.close()  # 창을 닫으면 closeEvent가 호출됨

    def add_exclude_dir(self):
        """제외할 디렉토리 추가"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "제외할 디렉토리 선택",
            str(Path.home())
        )
        if dir_path:
            # Vault 디렉토리 기준으로 상대 경로로 변환
            try:
                vault_dir = Path(self.vault_edit.text())
                rel_path = str(Path(dir_path).relative_to(vault_dir))
                if rel_path not in [self.exclude_list.item(i).text() for i in range(self.exclude_list.count())]:
                    self.exclude_list.addItem(rel_path)
            except ValueError:
                QMessageBox.warning(self, "경고", "Vault 디렉토리 내의 경로만 선택할 수 있습니다.")

    def remove_exclude_dir(self):
        """선택된 제외할 디렉토리 삭제"""
        current_item = self.exclude_list.currentItem()
        if current_item:
            self.exclude_list.takeItem(self.exclude_list.row(current_item)) 