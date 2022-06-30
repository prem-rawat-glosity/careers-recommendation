from pydantic import BaseSettings


class Settings(BaseSettings):
    api_base_url: str
    api_version: str
    service_type: str

    class Config:
        env_file = ".env"


settings = Settings()
