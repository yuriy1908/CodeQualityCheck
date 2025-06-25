from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from chardet import detect
import logging
import os
from datetime import datetime
import re
import warnings

# Фильтр всех предупреждений
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class CodeAnalyzer:
    def __init__(self, model_name, rag_db_path):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.generator = None  # Конвейер для генерации
        
        try:
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Инициализируем генератор как конвейер
            self._init_generator()
            
            # База знаний RAG (опционально)
            self.rag_db = self._init_rag_db(rag_db_path)
            
        except Exception as e:
            self.logger.critical(f"Init error: {str(e)}")
            raise

    def _init_generator(self):
        """Инициализация генератора как конвейера"""
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
            trust_remote_code=True
        )

    def _init_rag_db(self, path):
        if not path or not os.path.exists(path):
            return None
            
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            self.logger.warning(f"RAG init failed: {str(e)}")
            return None

    def analyze_file(self, file_path: str) -> str:
        try:
            if self._should_skip(file_path):
                return "ℹ️ File skipped (binary/system)"
                
            code = self._read_file_content(file_path)
            if not code:
                return "⚠️ Failed to read file"
                
            analysis = self.fast_generate_analysis(file_path, code)
            filename = os.path.basename(file_path)
            return self._postprocess_analysis(analysis, filename)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return f"⚠️ Analysis error: {str(e)}"


    def _should_skip(self, file_path):
        # Расширенный список пропускаемых расширений
        skip_exts = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.zip', '.rar',
            '.exe', '.dll', '.so', '.bin', '.pdf', '.ttf', '.woff', '.woff2',
            '.eot', '.otf', '.mp3', '.wav', '.mp4', '.avi', '.mov', '.doc',
            '.docx', '.xls', '.xlsx', '.pyc', '.gitignore', '.md', '.ini', '.txt',
            '.json', '.yaml', '.yml', '.log', '.lock', '.toml', '.cfg', '.conf'
        }
        
        # Пропускаем .git файлы и бинарные файлы
        if '/.git/' in file_path.replace('\\', '/'):
            return True
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext in skip_exts:
            return True
            
        return False

    def _read_file_content(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                raw = f.read()
                encoding = detect(raw)['encoding'] or 'utf-8'
                return raw.decode(encoding, errors='replace')
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def fast_generate_analysis(self, file_path, code):
        """Быстрая генерация анализа с использованием конвейера"""
        max_tokens = 3000
        encoded = self.tokenizer.encode(code)
        if len(encoded) > max_tokens:
            code = self.tokenizer.decode(
                encoded[:max_tokens],
                skip_special_tokens=True
            ) + "\n\n... [код усечен из-за длины]"
        
        filename = os.path.basename(file_path)
        
        # Упрощенный промпт с минимальными инструкциями
        prompt = (
            f"Анализируй код файла {filename} на проблемы безопасности, стиля, производительности и ошибок.\n"
            f"Если проблем нет, выведи 'Проблем не обнаружено'.\n"
            f"Если проблемы есть, выведи их в формате:\n"
            f"[строчка: проблемная строка]\n"
            f"[исправление: исправленная строка]\n\n"
            f"Код:\n{code}\n\n"
            f"Анализ:"
        )
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=400,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            response = result[0]['generated_text'].strip()
            # Извлекаем только текст после "Анализ:"
            if "Анализ:" in response:
                response = response.split("Анализ:", 1)[-1].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return f"⚠️ Ошибка генерации: {str(e)}"

    def _postprocess_analysis(self, analysis, filename):
        """Улучшенная постобработка анализа"""
        # Логируем сырой вывод для отладки
        self.logger.debug(f"Raw analysis for {filename}:\n{analysis}")
        
        # Шаблон для извлечения проблем
        problem_pattern = r"\[строчка:\s*(.+?)\]\s*\[исправление:\s*(.+?)\]"
        problems = re.findall(problem_pattern, analysis, re.DOTALL)
        
        if problems:
            formatted = []
            for problem in problems:
                problem_line = problem[0].strip()
                fix_line = problem[1].strip()
                
                # Очистка от лишних символов
                problem_line = re.sub(r'\s+', ' ', problem_line)
                fix_line = re.sub(r'\s+', ' ', fix_line)
                
                # Ограничение длины строк
                problem_line = problem_line[:200]
                fix_line = fix_line[:200]
                
                formatted.append(
                    f"[строчка: {problem_line}]\n"
                    f"[исправление: {fix_line}]"
                )
            
            return f"[{filename}]\n" + "\n\n".join(formatted)
        
        # Проверка на отсутствие проблем
        if re.search(r"Проблем\s+не\s+обнаружено", analysis, re.IGNORECASE):
            return f"[{filename}]\nПроблем не обнаружено"
        
        # Проверка на сообщения об ошибках
        if "Ошибка" in analysis or "⚠️" in analysis:
            return f"[{filename}]\n{analysis}"
        
        # Если вывод не пустой, но не соответствует формату
        if analysis.strip():
            return f"[{filename}]\n{analysis.strip()}"
        
        return f"[{filename}]\n⚠️ Не удалось извлечь результаты анализа"


    def generate_report(self, file_paths: list) -> str:
        """Генерация финального отчета"""
        final_report = "# Отчёт о качестве кода\n\n"
        valid_files = 0
        
        for path in file_paths:
            if not self._should_skip(path):
                clean_path = os.path.normpath(path).replace("\\", "/")
                analysis = self.analyze_file(path)
                
                if not analysis.strip():
                    analysis = "🤷 Анализ не дал результатов. Возможно, файл слишком сложен для автоматического анализа."
                
                if analysis.startswith("ℹ️") or analysis.startswith("⚠️"):
                    final_report += f"## Файл: `{clean_path}`\n{analysis}\n\n"
                    valid_files += 1
                    continue
                
                # Прямой вывод анализа без дополнительной обработки
                final_report += f"## Файл: `{clean_path}`\n{analysis}\n\n"
                valid_files += 1
                    
        if valid_files == 0:
            final_report += "⚠️ Нет файлов для анализа\n"
            final_report += "Возможные причины:\n"
            final_report += "- Все файлы пропущены как бинарные/системные\n"
            final_report += "- Ошибки чтения файлов\n"
            final_report += "- В репозитории нет файлов кода\n"
        
        # Сохранение отчета
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = "data/reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        final_path = os.path.join(reports_dir, f"report_{timestamp}.md")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_report)
            
        return final_path