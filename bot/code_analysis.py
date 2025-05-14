from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from chardet import detect
import logging
import os

class CodeAnalyzer:
    def __init__(self, model_name, rag_db_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = logging.getLogger(__name__)
        
        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.float16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            load_in_4bit=True if torch.cuda.is_available() else False
        )
        
        self.rag_db = self._load_rag_db(rag_db_path)

    def _load_rag_db(self, path):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    def _get_rag_context(self, code: str) -> str:
        try:
            docs = self.rag_db.similarity_search(code[:2000], k=2)
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            return f"Контекст недоступен: {str(e)}"

    def analyze_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                
                if b'\x00' in raw_data:
                    return "⚠️ Бинарный файл пропущен"
                
                encoding = detect(raw_data)['encoding'] or 'utf-8'
                # Увеличим лимит анализируемого кода
                code = raw_data.decode(encoding, errors='replace')[:15000]  # 15K символов
                
        except Exception as e:
            self.logger.error(f"Ошибка чтения {file_path}: {str(e)}")
            return f"⚠️ Ошибка чтения: {str(e)}"

        # Увеличим лимит контекста
        prompt = f"""Анализируй код (первые 15K символов):
    {code[:15000]}

Найди:
1. Критические ошибки
2. Потенциальные уязвимости
3. Стилевые проблемы
4. Рекомендации по оптимизации"""
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,  # Увеличим объем вывода
            temperature=0.7,
            top_p=0.9,
            do_sample=True
    )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_report(self, file_paths: list) -> str:
        report = "# Отчет о качестве кода\n\n"
        for path in file_paths:
            clean_path = os.path.normpath(path).replace("\\", "/")
            try:
                analysis = self.analyze_file(path)
                report += f"## Файл: `{clean_path}`\n{analysis}\n\n"
            except Exception as e:
                report += f"## Файл: `{clean_path}`\n⚠️ Ошибка: {str(e)}\n\n"
        return report