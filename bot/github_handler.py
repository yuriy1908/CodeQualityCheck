import os
import re
import logging
from datetime import datetime
from pathlib import Path
from git import Repo
from chardet import detect

logger = logging.getLogger(__name__)

def clone_repo(repo_url: str, base_path: str, user_id: int) -> str:
    try:
        repo_name = re.sub(r"[^\w\-]", "_", repo_url.split("/")[-1].replace(".git", ""))
        repo_path = Path(base_path) / f"{user_id}_{repo_name}"
        
        if repo_path.exists():
            return str(repo_path)

        Repo.clone_from(repo_url, str(repo_path))
        return str(repo_path)
    except Exception as e:
        logger.error(f"Ошибка клонирования: {str(e)}")
        raise RuntimeError(f"Ошибка клонирования: {str(e)}")

def analyze_repo(repo_path, analyzer):
    MAX_SIZE = 200000  # Увеличим до 200KB
    MAX_FILES = 100    # Увеличим лимит файлов
    
    valid_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) > MAX_SIZE:
                continue
            try:
                with open(file_path, 'rb') as f:
                    if b'\x00' in f.read(1024):
                        continue
                valid_files.append(file_path)
                if len(valid_files) >= MAX_FILES:
                    break
            except:
                continue
    # Остальной код без изменений
    
    report_content = analyzer.generate_report(valid_files[:MAX_FILES])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-]", "_", Path(repo_path).name)
    report_filename = f"report_{safe_name}_{timestamp}.md"
    report_path = os.path.join("data/reports", report_filename)
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    return report_path