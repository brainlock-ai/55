import torch
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import statistics
import os
import numpy as np
import json

# Create Base at module level so it can be imported
Base = declarative_base()

# Define the database model
class MinerHistory(Base):
    __tablename__ = 'miner_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(String, index=True, nullable=False)
    average_inference_per_second = Column(Float, nullable=False)
    prediction_match = Column(Float, nullable=False)
    score = Column(Float, nullable=True)
    stats = Column(String, nullable=True)  # JSON string containing response time, prediction match, etc.
    timestamp = Column(DateTime, default=datetime.now(timezone.utc), nullable=False, index=True)

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

    def pareto_score(self, inference_speed_and_accuracy_score):
        """
        Calculate Pareto-based score for response time.
        Score = (scale/x)^α where α is shape parameter and x being better as it decreases	
        This creates a heavy-tailed distribution favoring faster times.	
        Scores can exceed 1.0 for times below the scale factor.
        """
        # A higher speed and accuracy score is better, so x = 1/y here
        return pow(self.SCALE*inference_speed_and_accuracy_score, self.ALPHA)

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

    def calculate_score(self, inference_speed_and_accuracy_score: float, average_inference_per_second: float, average_cosine_similarity: float, hotkey: str) -> tuple[float, dict]:
        """
        Calculate score for the current response and store it in the database.
        Each response is scored immediately using the Pareto distribution.
        Historical statistics are calculated from up to 40 most recent entries.
        Stored in the database
        
        Args:
            inference_speed_and_accuracy_score (float): Average inference per second multiplied by the average cosine similarity
            average_inference_per_second (float): Average number computation or inference per second
            average_cosine_similarity (float): Average cosine similarity between the miner's outputs and the non-quantized non-fhe output	
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
            current_score = self.pareto_score(inference_speed_and_accuracy_score)

            # Create stats dictionary for current response
            current_stats = {
                "current_score": float(current_score),
                "average_inference_per_second": float(average_inference_per_second),
                "average_cosine_similarity": float(average_cosine_similarity)
            }

            # Add current response and its score to the database
            new_entry = MinerHistory(
                hotkey=hotkey,
                average_inference_per_second=average_inference_per_second,
                prediction_match=average_cosine_similarity,
                score=float(inference_speed_and_accuracy_score),
                stats=json.dumps(current_stats)
            )
            session.add(new_entry)
            session.commit()

            # Get historical statistics from up to 40 most recent entries
            stats_query = text("""
                WITH recent_history AS (
                    SELECT 
                        score,
                        stats::json as stats_json,
                        ROW_NUMBER() OVER (PARTITION BY hotkey ORDER BY timestamp DESC) as rn
                    FROM miner_history
                    WHERE hotkey = :hotkey
                    AND timestamp >= NOW() - INTERVAL '6 hours'
                    AND stats IS NOT NULL
                ),
                last_40_entries AS (
                    SELECT *
                    FROM recent_history
                    WHERE rn <= 40
                ),
                response_times AS (
                    SELECT 
                        (stats_json->>'average_inference_per_second')::float as average_inference_per_second,
                        score,
                        (stats_json->>'average_cosine_similarity')::boolean as average_cosine_similarity
                    FROM last_40_entries
                )
                SELECT 
                    AVG(average_inference_per_second) as rt_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY average_inference_per_second) as rt_median,
                    STDDEV(average_inference_per_second) as rt_std,
                    AVG(score) as score_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as score_median,
                    STDDEV(score) as score_std,
                    COUNT(*) as entry_count
                FROM response_times;
            """)
            
            result = session.execute(stats_query, {"hotkey": hotkey}).fetchone()
            
            # Prepare aggregated stats
            stats = {
                "current_score": float(current_score),
                "average_inference_per_second": float(average_inference_per_second),
                "average_cosine_similarity": bool(average_cosine_similarity),
                "response_time_stats": (
                    float(result.rt_mean if result.rt_mean is not None else average_inference_per_second),
                    float(result.rt_median if result.rt_median is not None else average_inference_per_second),
                    float(result.rt_std if result.rt_std is not None else 0.0)
                ),
                "score_stats": (
                    float(result.score_mean if result.score_mean is not None else current_score),
                    float(result.score_median if result.score_median is not None else current_score),
                    float(result.score_std if result.score_std is not None else 0.0)
                ),
                "entry_count": int(result.entry_count if result.entry_count is not None else 1)
            }

            return current_score, stats

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
