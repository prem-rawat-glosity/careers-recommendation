from pydantic import BaseSettings


class Settings(BaseSettings):
    FLASK_APP: str = "flask_api"
    FLASK_ENV: str = "development"
    FLASK_RUN_HOST: str = "127.0.0.1"
    FLASK_RUN_PORT: str = "8000"
    FLASK_RUN_EXTRA_FILES: str = "config.yml"
    api_base_url: str = "https://qaapi.glosity.club/api"
    api_version: str = "v1"
    service_type: str = "mobile"
    models_base_url: str = "https://glosity.s3.ap-south-1.amazonaws.com/AI/careers"

    class Config:
        env_file = ".env"


settings = Settings()
