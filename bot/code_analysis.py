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
                
            return self._fast_generate_analysis(file_path, code)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return f"⚠️ Analysis error: {str(e)}"

    def _should_skip(self, file_path):
        # Пропускаем .git файлы и бинарные файлы
        return '/.git/' in file_path.replace('\\', '/')

    def _read_file_content(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                raw = f.read()
                encoding = detect(raw)['encoding'] or 'utf-8'
                return raw.decode(encoding, errors='replace')
        except:
            return None

    def _fast_generate_analysis(self, file_path, code):
        """Быстрая генерация анализа с использованием конвейера"""
        prompt = f"""Analyze this code from {os.path.basename(file_path)}:
{code}

Provide concise analysis covering:
1. Critical errors
2. Security issues
3. Style violations
4. Optimization tips
5. Overall quality"""

        try:
            # Генерация ответа с ограничением длины
            result = self.generator(
                prompt,
                max_new_tokens=300,
                temperature=0.3,  # Понижаем "креативность"
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_sequence="###"  # Явное стоп-слово
            )
            
            # Извлекаем сгенерированный текст
            response = result[0]['generated_text']
            
            # Удаляем промпт из ответа
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return f"⚠️ Generation error: {str(e)}"
    
    def _translate_to_russian(self, text: str) -> str:
        """Перевод текста на русский с помощью конвейера"""
        if not text.strip() or len(text) < 50:
            return text
            
        try:
            # Промпт для перевода с сохранением форматирования
            prompt = f"""Переведи следующий технический анализ на русский язык, сохраняя Markdown форматирование и технические термины:

{text}

Перевод:"""
            
            # Генерация перевода
            result = self.generator(
                prompt,
                max_new_tokens=len(text) + 100,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Извлекаем перевод
            translation = result[0]['generated_text'].replace(prompt, "").strip()
            return translation
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return text  # Возвращаем оригинал в случае ошибки

    def _clean_analysis(self, analysis: str) -> str:
        # Удаляем всё до последнего вхождения "ОТВЕТ" или "ANSWER"
        for marker in ["### ОТВЕТ", "### ANSWER", "Ответ:", "Analysis:"]:
            idx = analysis.rfind(marker)  # Ищем последнее вхождение
            if idx != -1:
                analysis = analysis[idx + len(marker):]
        
        # Дополнительная очистка
        analysis = analysis.split("###")[0]  # Удаляем всё после разделителя
        analysis = analysis.split("```")[0]  # Удаляем Markdown-блоки
        return analysis.strip()

    def generate_report(self, file_paths: list) -> str:
        """Генерация финального отчета на русском"""
        final_report = "# Отчёт о качестве кода\n\n"
        valid_files = 0
        
        for path in file_paths:
            if not self._should_skip(path):
                clean_path = os.path.normpath(path).replace("\\", "/")
                analysis = self.analyze_file(path)
                
                # Для информационных сообщений просто добавляем в отчет
                if analysis.startswith("ℹ️") or analysis.startswith("⚠️"):
                    final_report += f"## Файл: `{clean_path}`\n{analysis}\n\n"
                    valid_files += 1
                    continue
                
                try:
                    # Очищаем анализ
                    clean_analysis = self._clean_analysis(analysis)
                    
                    # Переводим на русский
                    ru_analysis = self._translate_to_russian(clean_analysis)
                    
                    final_report += f"## Файл: `{clean_path}`\n{ru_analysis}\n\n"
                    valid_files += 1
                except Exception as e:
                    final_report += f"## Файл: `{clean_path}`\n⚠️ Ошибка обработки: {str(e)}\n\n"
                
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