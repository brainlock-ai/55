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

    def pareto_score(self, time):
        """Calculate Pareto-based score for response time."""
        return pow(self.SCALE / time, self.ALPHA)

    def calculate_score(self, response_time: float, predictions_match: bool, hotkey: str) -> tuple[float, dict]:
        """Calculate score for the current response and store it in the database."""
        if not hotkey:
            raise ValueError("hotkey is required and cannot be None or empty")

        session = self.Session()

        try:
            # Score this specific response using Pareto distribution
            current_score = self.pareto_score(response_time) if predictions_match else 0.0

            # Create stats dictionary for current response
            current_stats = {
                "current_score": float(current_score),
                "response_time": float(response_time),
                "predictions_match": bool(predictions_match),
                "failure_rate": float(0.0 if predictions_match else 1.0)
            }

            # Add current response and its score to the database
            new_entry = MinerHistory(
                hotkey=hotkey,
                score=float(current_score),
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
                        (stats_json->>'response_time')::float as response_time,
                        score,
                        (stats_json->>'predictions_match')::boolean as predictions_match
                    FROM last_40_entries
                )
                SELECT 
                    AVG(response_time) as rt_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time) as rt_median,
                    STDDEV(response_time) as rt_std,
                    AVG(score) as score_mean,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as score_median,
                    STDDEV(score) as score_std,
                    1.0 - (SUM(CASE WHEN predictions_match THEN 1 ELSE 0 END)::float / COUNT(*)) as failure_rate,
                    COUNT(*) as entry_count
                FROM response_times;
            """)
            
            result = session.execute(stats_query, {"hotkey": hotkey}).fetchone()
            
            # Prepare aggregated stats
            stats = {
                "current_score": float(current_score),
                "response_time": float(response_time),
                "predictions_match": bool(predictions_match),
                "response_time_stats": (
                    float(result.rt_mean if result.rt_mean is not None else response_time),
                    float(result.rt_median if result.rt_median is not None else response_time),
                    float(result.rt_std if result.rt_std is not None else 0.0)
                ),
                "score_stats": (
                    float(result.score_mean if result.score_mean is not None else current_score),
                    float(result.score_median if result.score_median is not None else current_score),
                    float(result.score_std if result.score_std is not None else 0.0)
                ),
                "failure_rate": float(result.failure_rate if result.failure_rate is not None else (0.0 if predictions_match else 1.0)),
                "entry_count": int(result.entry_count if result.entry_count is not None else 1)
            }

            return current_score, stats

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
