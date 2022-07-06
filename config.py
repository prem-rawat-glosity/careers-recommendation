from pydantic import BaseSettings


class Settings(BaseSettings):
    api_base_url: str = "https://qaapi.glosity.club/api"
    api_version: str = "v1"
    service_type: str = "mobile"
    models_base_url: str = "https://glosity.s3.ap-south-1.amazonaws.com/AI/careers"

    class Config:
        env_file = ".env"


settings = Settings()
