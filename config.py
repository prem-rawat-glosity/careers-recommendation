from pydantic import BaseSettings


class Settings(BaseSettings):
    FLASK_APP: str = 'FLASK_APP'
    FLASK_ENV: str = 'FLASK_ENV'
    FLASK_RUN_HOST: str = 'FLASK_RUN_HOST'
    FLASK_RUN_PORT: str = 'FLASK_RUN_PORT'
    FLASK_RUN_EXTRA_FILES: str = 'FLASK_RUN_EXTRA_FILES'
    api_base_url: str = 'API_BASE_URL'
    api_version: str = 'API_VERSION'
    service_type: str = 'SERVICE_TYPE'
    models_base_url: str = 'CAREERS_MODEL_BASE_URL'

    class Config:
        env_file = ".env"


settings = Settings()
