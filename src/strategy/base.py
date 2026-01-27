from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data: pd.DataFrame) -> int:
        """
        Returns:
        1 (Long)
        0 (Neutral)
        -1 (Short)
        """
        pass
