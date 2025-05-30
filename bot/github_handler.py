import os
import re
import logging
from pathlib import Path
from git import Repo

logger = logging.getLogger(__name__)

def clone_repo(repo_url: str, base_path: str, user_id: int) -> str:
    try:
        repo_name = re.sub(r"[^\w\-]", "_", repo_url.split("/")[-1].replace(".git", ""))
        repo_path = Path(base_path) / f"{user_id}_{repo_name}"
        
        if repo_path.exists():
            return str(repo_path)

        # Неглубокое клонирование
        Repo.clone_from(repo_url, str(repo_path), depth=1)
        return str(repo_path)
    except Exception as e:
        logger.error(f"Cloning error: {str(e)}")
        raise RuntimeError(f"Cloning error: {str(e)}")

def analyze_repo(repo_path, analyzer):
    """Анализ всех файлов в репозитории без ограничений"""
    valid_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Пропускаем только .git файлы
            if '/.git/' in file_path.replace('\\', '/'):
                continue
                
            valid_files.append(file_path)
    
    return analyzer.generate_report(valid_files)