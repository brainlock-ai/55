import torch
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, inspect, text
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

    def calculate_ips_score(self, inference_per_second: float, predictions_match: bool, hotkey: str) -> tuple[float, dict]:
        """
        Calculate inference per second score for the current response and store it in the database.
        Each response is scored immediately using the Pareto distribution.
        Historical statistics are calculated from up to 40 most recent entries.
        
        Args:
            response_time (float): Time taken for the computation	
            predictions_match (bool): Whether predictions matched the original model	
            hotkey (str): The miner's hotkey for tracking history (required)	
        
        Returns:	
            tuple[float, dict]: (current_score, statistics_dict)	
            - current_score: Score for this specific response (>= 0)
            - statistics_dict: Historical statistics from up to 40 most recent entries
        """
        if not hotkey:
            raise ValueError("hotkey is required and cannot be None or empty")

        session = self.Session()

        try:
            # Score this specific response using Pareto distribution
            current_score = self.pareto_score(response_time) if predictions_match else 0.0

            # Add current response and its score to the database
            new_entry = MinerHistory(
                hotkey=hotkey, 
                response_time=response_time, 
                prediction_match=predictions_match,
                score=current_score  # Store the score for this specific response
            )
            session.add(new_entry)
            session.commit()

            # Get historical statistics from up to 40 most recent entries (including this one)
            stats_query = text("""
                WITH recent_history AS (
                    SELECT 
                        response_time,
                        prediction_match,
                        score,
                        ROW_NUMBER() OVER (PARTITION BY hotkey ORDER BY timestamp DESC) as rn
                    FROM miner_history
                    WHERE hotkey = :hotkey
                ),
                last_40_entries AS (
                    SELECT *
                    FROM recent_history
                    WHERE rn <= 40
                )
                SELECT 
                    AVG(response_time) as rt_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time) as rt_median,
                    STDDEV(response_time) as rt_std,
                    AVG(score) as score_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as score_median,
                    STDDEV(score) as score_std,
                    1.0 - (SUM(CASE WHEN prediction_match THEN 1 ELSE 0 END)::float / COUNT(*)) as failure_rate,
                    COUNT(*) as entry_count
                FROM last_40_entries;
            """)
            
            result = session.execute(stats_query, {"hotkey": hotkey}).fetchone()
            
            stats = {
                "current_score": current_score,  # Add the current response's score to stats
                "response_time_stats": (
                    result.rt_mean if result.rt_mean is not None else 0.0,
                    result.rt_median if result.rt_median is not None else 0.0,
                    result.rt_std if result.rt_std is not None else 0.0
                ),
                "score_stats": (
                    result.score_mean if result.score_mean is not None else 0.0,
                    result.score_median if result.score_median is not None else 0.0,
                    result.score_std if result.score_std is not None else 0.0
                ),
                "failure_rate": result.failure_rate if result.failure_rate is not None else 0.0,
                "entry_count": result.entry_count if result.entry_count is not None else 1
            }

            return current_score, stats

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


    def calculate_score(self, response_time: float, predictions_match: bool, hotkey: str) -> tuple[float, dict]:
        """
        Deprecated
        Calculate score for the current response and store it in the database.
        Each response is scored immediately using the Pareto distribution.
        Historical statistics are calculated from up to 40 most recent entries.
        
        Args:	
            response_time (float): Time taken for the computation	
            predictions_match (bool): Whether predictions matched the original model	
            hotkey (str): The miner's hotkey for tracking history (required)	
            	
        Returns:	
            tuple[float, dict]: (current_score, statistics_dict)	
            - current_score: Score for this specific response (>= 0)
            - statistics_dict: Historical statistics from up to 40 most recent entries
        """
        if not hotkey:
            raise ValueError("hotkey is required and cannot be None or empty")

        session = self.Session()

        try:
            # Score this specific response using Pareto distribution
            current_score = self.pareto_score(response_time) if predictions_match else 0.0

            # Add current response and its score to the database
            new_entry = MinerHistory(
                hotkey=hotkey, 
                response_time=response_time, 
                prediction_match=predictions_match,
                score=current_score  # Store the score for this specific response
            )
            session.add(new_entry)
            session.commit()

            # Get historical statistics from up to 40 most recent entries (including this one)
            stats_query = text("""
                WITH recent_history AS (
                    SELECT 
                        response_time,
                        prediction_match,
                        score,
                        ROW_NUMBER() OVER (PARTITION BY hotkey ORDER BY timestamp DESC) as rn
                    FROM miner_history
                    WHERE hotkey = :hotkey
                ),
                last_40_entries AS (
                    SELECT *
                    FROM recent_history
                    WHERE rn <= 40
                )
                SELECT 
                    AVG(response_time) as rt_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time) as rt_median,
                    STDDEV(response_time) as rt_std,
                    AVG(score) as score_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as score_median,
                    STDDEV(score) as score_std,
                    1.0 - (SUM(CASE WHEN prediction_match THEN 1 ELSE 0 END)::float / COUNT(*)) as failure_rate,
                    COUNT(*) as entry_count
                FROM last_40_entries;
            """)
            
            result = session.execute(stats_query, {"hotkey": hotkey}).fetchone()
            
            stats = {
                "current_score": current_score,  # Add the current response's score to stats
                "response_time_stats": (
                    result.rt_mean if result.rt_mean is not None else 0.0,
                    result.rt_median if result.rt_median is not None else 0.0,
                    result.rt_std if result.rt_std is not None else 0.0
                ),
                "score_stats": (
                    result.score_mean if result.score_mean is not None else 0.0,
                    result.score_median if result.score_median is not None else 0.0,
                    result.score_std if result.score_std is not None else 0.0
                ),
                "failure_rate": result.failure_rate if result.failure_rate is not None else 0.0,
                "entry_count": result.entry_count if result.entry_count is not None else 1
            }

            return current_score, stats

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
