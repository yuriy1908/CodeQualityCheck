import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from bot.github_handler import clone_repo, analyze_repo
from bot.code_analysis import CodeAnalyzer
from config import Config

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class CodeQualityBot:
    def __init__(self):
        # Отложенная инициализация анализатора
        self.analyzer = None
        self.logger = logging.getLogger(__name__)
        self.application = None

    async def init_analyzer(self):
        """Асинхронная инициализация анализатора"""
        if not self.analyzer:
            self.analyzer = CodeAnalyzer(Config.MODEL_NAME, Config.RAG_DB_PATH)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        try:
            await update.message.reply_text(
                "🚀 Привет! Я бот для анализа качества кода.\n"
                "Доступные команды:\n"
                "/connect <URL-репозитория> - подключить репозиторий\n"
                "/analyze - начать анализ кода\n"
                "/stop - остановить бота"
            )
        except Exception as e:
            self.logger.error(f"Ошибка в /start: {str(e)}")

    async def connect_repo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /connect"""
        try:
            if not context.args:
                await update.message.reply_text("❌ Укажите URL репозитория")
                return

            repo_url = context.args[0]
            user_id = update.effective_user.id
            repo_path = clone_repo(repo_url, Config.REPOS_ROOT, user_id)
            
            context.user_data["repo_path"] = repo_path
            await update.message.reply_text(
                f"✅ Репозиторий подключен: {os.path.basename(repo_path)}"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")
            self.logger.error(f"Ошибка подключения репа: {str(e)}")

    async def analyze_code(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if "repo_path" not in context.user_data:
                await update.message.reply_text("❌ Сначала подключите репозиторий")
                return

            await self.init_analyzer()  # Инициализируем анализатор
            repo_path = context.user_data["repo_path"]
            
            await update.message.reply_text(
                "🔍 Начинаю анализ кода..."
            )

            # Асинхронный анализ
            report_path = await asyncio.get_event_loop().run_in_executor(
                None,
                analyze_repo,
                repo_path,
                self.analyzer
            )

            await update.message.reply_document(
                document=open(report_path, "rb"),
                caption="📊 Отчет готов!",
                filename="code_quality_report.md"
            )

        except Exception as e:
            await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")
            self.logger.error(f"Ошибка анализа: {str(e)}")

    async def stop_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stop"""
        try:
            await update.message.reply_text("🛑 Останавливаю бота...")
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            os._exit(0)
        except Exception as e:
            self.logger.error(f"Ошибка остановки: {str(e)}")

def main():
    """Запуск бота"""
    try:
        bot = CodeQualityBot()
        application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        bot.application = application

        # Регистрация обработчиков
        handlers = [
            CommandHandler("start", bot.start),
            CommandHandler("connect", bot.connect_repo),
            CommandHandler("analyze", bot.analyze_code),
            CommandHandler("stop", bot.stop_bot)
        ]
        
        for handler in handlers:
            application.add_handler(handler)

        application.run_polling(
            poll_interval=2.0,
            timeout=100,
            drop_pending_updates=True
        )
    except Exception as e:
        logging.critical(f"Фатальная ошибка: {str(e)}")

if __name__ == "__main__":
    main()