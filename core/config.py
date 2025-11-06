from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    data_raw_dir: Path = Path("data/raw")
    data_cache_dir: Path = Path("data/cache")
    default_start: str = "2000-01-01"
    default_freq: str = "1wk"
    tz: str = "UTC"

settings = Settings()
