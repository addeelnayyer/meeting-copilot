"""Tests never read the developer's .env (loaded when copilot.config is imported), so they see CI's defaults."""

import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False
