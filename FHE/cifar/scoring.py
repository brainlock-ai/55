import torch
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import statistics
import os
import numpy as np

# Create Base at module level so it can be imported
Base = declarative_base()

# Define the database model
class MinerHistory(Base):
    __tablename__ = 'miner_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(String, index=True, nullable=False)
    response_time = Column(Float, nullable=False)
    prediction_match = Column(Boolean, nullable=False)
    score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)

def create_tables_if_not_exist(engine):
    """Create tables only if they don't exist."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if 'miner_history' not in tables:
        Base.metadata.create_all(engine)
        return True
    return False

# SimplifiedReward class
class SimplifiedReward:
    """
    This is a simplified version, where vTrust should be more stable. We will work on
    using a ranging "scale" factor based on the top k % of miner's response times
    """
    def __init__(self, db_url=None):
        if db_url is None:
            db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'miner_data')}"
        
        # Set up the database
        self.engine = create_engine(db_url)
        create_tables_if_not_exist(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Thresholds and constants
        self.FAILURE_THRESHOLD = 0.6 # 20% failure threshold
        self.EXPONENTIAL_BASE = 2.5
        self.ALPHA = 3.0  # Shape parameter for Pareto distribution
        self.SCALE = 40.0  # Scale factor to normalize scores

    def pareto_score(self, time):
        """
        Calculate Pareto-based score for response time.
        Score = (scale/x)^α where α is shape parameter.	
        This creates a heavy-tailed distribution favoring faster times.	
        Scores can exceed 1.0 for times below the scale factor.
        """
        return pow(self.SCALE / time, self.ALPHA)

    def _calculate_stats(self, values):
        """Calculate mean, median, and standard deviation of a sequence"""
        if not values:
            return 0, 0, 0

        values_array = np.array(values)
        return (
            np.mean(values_array),
            np.median(values_array),
            np.std(values_array) if len(values_array) > 1 else 0
        )

    def calculate_score(self, response_time: float, predictions_match: bool, hotkey: str) -> tuple[float, dict]:
        """
        Calculate score based on historical performance and current response.
        Score is based on median response time (lower is better) with failure rate penalties.	
        Uses Pareto distribution for time scoring to heavily favor faster responses.	
        Scores can exceed 1.0 for response times below 40s.	
        	
        Args:	
            response_time (float): Time taken for the computation	
            predictions_match (bool): Whether predictions matched the original model	
            hotkey (str): The miner's hotkey for tracking history (required)	
            	
        Returns:	
            tuple[float, dict]: (score, statistics_dict)	
            - score: Score >= 0, can exceed 1.0 for very fast responses	
            - statistics_dict: Dictionary containing computed statistics
        """
        if not hotkey:
            raise ValueError("hotkey is required and cannot be None or empty")

        session = self.Session()

        # Add current response to the database
        new_entry = MinerHistory(hotkey=hotkey, response_time=response_time, prediction_match=predictions_match)
        session.add(new_entry)
        session.commit()

        # Retrieve history for this miner
        history = session.query(MinerHistory).filter_by(hotkey=hotkey).all()
        response_times = [entry.response_time for entry in history]
        predictions = [entry.prediction_match for entry in history]
        scores = [entry.score for entry in history if entry.score is not None]

        # Calculate statistics
        rt_stats = self._calculate_stats(response_times)

        # If not enough history, use current response time for stats
        if len(predictions) < 2:
            initial_score = self.pareto_score(response_time) if predictions_match else 0.0
            new_entry.score = initial_score
            session.commit()
            session.close()
            return initial_score, {
                "response_time_stats": (response_time, response_time, 0),
                "score_stats": (initial_score, initial_score, 0),
                "failure_rate": 0 if predictions_match else 1
            }

        # Calculate failure rate
        failure_rate = 1.0 - (sum(predictions) / len(predictions))

        # Get response times of correct predictions
        correct_times = [t for t, p in zip(response_times, predictions) if p]
        if not correct_times:
            session.close()
            return 0.0, {
                "response_time_stats": rt_stats,
                "score_stats": (0, 0, 0),
                "failure_rate": failure_rate
            }

        median_response_time = statistics.median(correct_times)

        # Calculate Pareto-based score for response time
        base_score = self.pareto_score(median_response_time)

        # # Apply failure penalty if above threshold
        # if failure_rate > self.FAILURE_THRESHOLD:
        #     excess_failure = failure_rate - self.FAILURE_THRESHOLD
        #     penalty = pow(self.EXPONENTIAL_BASE, (excess_failure * 100)) - 1.0
        #     base_score = base_score / (1.0 + penalty)

        final_score = base_score  # Removed max(0.0, base_score) since we're not applying penalties

        # Update score in the database
        new_entry.score = final_score
        session.commit()
        session.close()

        stats = {
            "response_time_stats": rt_stats,
            "score_stats": self._calculate_stats(scores + [final_score]),
            "failure_rate": failure_rate
        }

        return final_score, stats
