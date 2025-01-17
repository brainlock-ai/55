from prometheus_client import start_http_server, Gauge, Counter
from sqlalchemy import create_engine, text
import time
import os
import logging
from datetime import datetime, timedelta
from scoring import Base, create_tables_if_not_exist  # Import the safe table creation function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgresExporter:
    def __init__(self, port=6969):
        # Get database credentials from environment variables
        self.db_user = os.getenv('POSTGRES_USER')
        self.db_password = os.getenv('POSTGRES_PASSWORD')
        self.db_name = os.getenv('POSTGRES_DB', 'miner_data')
        self.db_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.db_port = os.getenv('POSTGRES_PORT', '5432')
        
        # Initialize Prometheus metrics based on available DB fields
        self.validation_requests = Counter(
            'validator_validation_requests_total',
            'Total number of validation requests made',
            ['status', 'hotkey']
        )
        
        self.miner_scores = Gauge(
            'validator_miner_scores',
            'Current scores of miners',
            ['hotkey']
        )
        
        self.prediction_accuracy = Gauge(
            'validator_prediction_accuracy',
            'Accuracy of predictions',
            ['hotkey']
        )
        
        self.miner_response_time = Gauge(
            'validator_miner_response_time',
            'Latest response time from miner',
            ['hotkey']
        )
        
        self.miner_avg_score = Gauge(
            'validator_miner_rolling_avg_score',
            'Average score over last 40 responses',
            ['hotkey']
        )
        
        self.response_time_mean = Gauge(
            'validator_miner_response_time_mean',
            'Mean response time over last 40 responses',
            ['hotkey']
        )
        
        self.response_time_std = Gauge(
            'validator_miner_response_time_std',
            'Standard deviation of response time over last 40 responses',
            ['hotkey']
        )
        
        self.response_time_median = Gauge(
            'validator_miner_response_time_median',
            'Median response time over last 40 responses',
            ['hotkey']
        )
        
        self.active_miners = Gauge(
            'validator_active_miners',
            'Number of active miners in the network'
        )

        #self.failure_rate = Gauge(
        #    'validator_miner_failure_rate',
        #    'Failure rate of miner predictions',
        #    ['hotkey']
        #)

        # Initialize database connection
        self.engine = create_engine(
            f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        )
        
        # Create tables if they don't exist
        try:
            if create_tables_if_not_exist(self.engine):
                logger.info("Database tables initialized successfully")
            else:
                logger.info("Database tables already exist")
        except Exception as e:
            logger.error(f"Error checking/initializing database tables: {str(e)}")
        
        # Start the server
        start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")

    def collect_metrics(self):
        """Collect metrics from the database and update Prometheus gauges"""
        try:
            with self.engine.connect() as conn:
                # Get active miners (miners seen in the last hour)
                active_miners_query = text("""
                    SELECT COUNT(DISTINCT hotkey) 
                    FROM miner_history 
                    WHERE timestamp >= NOW() - INTERVAL '6 hours'
                """)
                active_count = conn.execute(active_miners_query).scalar() or 0
                self.active_miners.set(active_count)

                # Get per-miner statistics using only available fields
                miner_stats_query = text("""
                    WITH recent_records AS (
                        SELECT 
                            hotkey,
                            score,
                            stats::json as stats_json,
                            timestamp,
                            ROW_NUMBER() OVER (PARTITION BY hotkey ORDER BY timestamp DESC) as rn
                        FROM miner_history
                        WHERE timestamp >= NOW() - INTERVAL '6 hours'
                        AND stats IS NOT NULL
                    ),
                    miner_aggregates AS (
                        SELECT 
                            hotkey,
                            AVG(score) as avg_score,
                            COUNT(*) as total_requests,
                            AVG((stats_json->>'average_inference_per_second')::float) as avg_response_time,
                            STDDEV((stats_json->>'average_inference_per_second')::float) as std_response_time,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (stats_json->>'average_inference_per_second')::float) as median_response_time
                        FROM recent_records
                        WHERE rn <= 40  -- Keep last 40 records
                        GROUP BY hotkey
                    ),
                    latest_values AS (
                        SELECT 
                            hotkey,
                            (stats_json->>'average_inference_per_second')::float as latest_response_time,
                            score as latest_score,
                            (stats_json->>'average_cosine_similarity')::boolean as latest_predictions_match
                        FROM recent_records
                        WHERE rn = 1
                    )
                    SELECT 
                        a.*,
                        l.latest_response_time,
                        l.latest_score,
                        l.latest_predictions_match
                    FROM miner_aggregates a
                    JOIN latest_values l ON a.hotkey = l.hotkey
                """)

                results = conn.execute(miner_stats_query)
                
                for row in results:
                    hotkey = row.hotkey
                    
                    # Update all metrics for this miner with NULL checks
                    if row.latest_score is not None:
                        self.miner_scores.labels(hotkey=hotkey).set(row.latest_score)
                    #if row.failure_rate is not None:
                    #    prediction_accuracy = 1.0 - row.failure_rate
                    #    self.prediction_accuracy.labels(hotkey=hotkey).set(prediction_accuracy)
                    if row.latest_response_time is not None:
                        self.miner_response_time.labels(hotkey=hotkey).set(row.latest_response_time)
                    if row.avg_score is not None:
                        self.miner_avg_score.labels(hotkey=hotkey).set(row.avg_score)
                    if row.avg_response_time is not None:
                        self.response_time_mean.labels(hotkey=hotkey).set(row.avg_response_time)
                    if row.std_response_time is not None:
                        self.response_time_std.labels(hotkey=hotkey).set(row.std_response_time)
                    if row.median_response_time is not None:
                        self.response_time_median.labels(hotkey=hotkey).set(row.median_response_time)
                    #if row.failure_rate is not None:
                    #    self.failure_rate.labels(hotkey=hotkey).set(row.failure_rate)
                    
                    # Update request counters only if we have valid counts
                    #if row.total_requests is not None:
                    #    success_count = int(row.total_requests * (1.0 - row.failure_rate))
                    #    failure_count = row.total_requests - success_count
                        
                        # Set the counter values directly
                    #    self.validation_requests.labels(status='success', hotkey=hotkey)._value.set(success_count)
                    #    self.validation_requests.labels(status='failure', hotkey=hotkey)._value.set(failure_count)

        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")

    def run_metrics_loop(self):
        """Run the metrics collection loop."""
        while True:
            try:
                self.collect_metrics()
            except Exception as e:
                logger.error(f"Error in metrics loop: {str(e)}")
            time.sleep(15)

def main():
    exporter = PostgresExporter()
    exporter.run_metrics_loop()

if __name__ == "__main__":
    main() 