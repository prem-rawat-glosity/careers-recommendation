from pydantic import BaseSettings


class Settings(BaseSettings):
    FLASK_APP: str
    FLASK_ENV: str
    FLASK_RUN_HOST: str
    FLASK_RUN_PORT: str
    FLASK_RUN_EXTRA_FILES: str
    API_BASE_URL: str
    API_VERSION: str
    SERVICE_TYPE: str
    CAREERS_MODEL_BASE_URL: str

    class Config:
        env_file = ".env"


settings = Settings()
