from functools import wraps, partial
from dataclasses import dataclass
import time

from src.settings import set_logging


@dataclass
class Utils:
    logger = set_logging()
    def log(self,log_info:str="",func=None):
        if func is None:
            return partial(self.log, log_info)
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_ = time.time()
            result= func(*args, **kwargs)
            end_ = time.time()
            if log_info=="":
                self.logger.info(f"Time elapsed {end_- start_:.3f}")
            else:
                self.logger.info(f"Time elapsed {end_- start_:.3f} - {log_info}")
            return result
        return wrapper


if __name__=="__main__":
    __all__ = ["Timer"]


