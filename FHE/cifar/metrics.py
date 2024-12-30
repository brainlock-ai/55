from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY
import time
from collections import defaultdict, deque
import numpy as np

class ValidatorMetrics:
    def __init__(self, port=6969, start_server=True):
        # Initialize response time tracking
        self.response_times = defaultdict(lambda: deque(maxlen=40))
        self.scores = defaultdict(lambda: deque(maxlen=40))
        
        # Initialize Prometheus metrics
        self.validation_requests = Counter(
            'validator_validation_requests_total',
            'Total number of validation requests made',
            ['status', 'hotkey'],
            registry=REGISTRY
        )
        
        self.validation_latency = Histogram(
            'validator_validation_latency_seconds',
            'Time spent validating miners',
            ['hotkey'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=REGISTRY
        )
        
        self.active_miners = Gauge(
            'validator_active_miners',
            'Number of active miners in the network',
            registry=REGISTRY
        )
        
        self.miner_scores = Gauge(
            'validator_miner_scores',
            'Current scores of miners',
            ['hotkey'],
            registry=REGISTRY
        )
        
        self.prediction_accuracy = Gauge(
            'validator_prediction_accuracy',
            'Accuracy of predictions',
            ['hotkey'],
            registry=REGISTRY
        )

        # Add new metrics for better miner tracking
        self.miner_last_seen = Gauge(
            'validator_miner_last_seen',
            'Timestamp when miner was last seen',
            ['hotkey'],
            registry=REGISTRY
        )

        self.miner_response_time = Gauge(
            'validator_miner_response_time',
            'Latest response time from miner',
            ['hotkey'],
            registry=REGISTRY
        )

        # New metrics for aggregated statistics
        self.miner_median_response_time = Gauge(
            'validator_miner_median_response_time',
            'Median response time over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )

        self.miner_avg_accuracy = Gauge(
            'validator_miner_avg_accuracy',
            'Average accuracy over all responses',
            ['hotkey'],
            registry=REGISTRY
        )

        self.miner_avg_score = Gauge(
            'validator_miner_rolling_avg_score',
            'Average score over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )

        # Add new metrics for per-request statistics
        self.response_time_mean = Gauge(
            'validator_miner_response_time_mean',
            'Mean response time over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )
        
        self.response_time_std = Gauge(
            'validator_miner_response_time_std',
            'Standard deviation of response time over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )
        
        self.response_time_median = Gauge(
            'validator_miner_response_time_median',
            'Median response time over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )

        # Similar statistics for scores
        self.score_mean = Gauge(
            'validator_miner_score_mean',
            'Mean score over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )
        
        self.score_std = Gauge(
            'validator_miner_score_std',
            'Standard deviation of score over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )
        
        self.score_median = Gauge(
            'validator_miner_score_median',
            'Median score over last 40 responses',
            ['hotkey'],
            registry=REGISTRY
        )

        if start_server:
            self.start_server(port)

    def start_server(self, port):
        """Start the Prometheus metrics server"""
        start_http_server(port)
        print(f"Started Prometheus metrics server on port {port}")

    def _calculate_median(self, values):
        """Helper to calculate median of a sequence"""
        sorted_values = sorted(values)
        length = len(sorted_values)
        if length == 0:
            return 0
        if length % 2 == 0:
            return (sorted_values[length//2 - 1] + sorted_values[length//2]) / 2
        return sorted_values[length//2]

    def _calculate_mean(self, values):
        """Helper to calculate mean of a sequence"""
        return sum(values) / len(values) if values else 0

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

    def record_validation_attempt(self, hotkey: str, success: bool, duration: float):
        """Helper method to record a validation attempt with all related metrics"""
        status = 'success' if success else 'failure'
        self.validation_requests.labels(status=status, hotkey=hotkey).inc()
        self.validation_latency.labels(hotkey=hotkey).observe(duration)
        self.miner_last_seen.labels(hotkey=hotkey).set(time.time())
        self.miner_response_time.labels(hotkey=hotkey).set(duration)
        
        # Only track response times for successful validations
        if success:
            self.response_times[hotkey].append(duration)
            
            # Calculate and update response time statistics
            mean, median, std = self._calculate_stats(self.response_times[hotkey])
            self.response_time_mean.labels(hotkey=hotkey).set(mean)
            self.response_time_median.labels(hotkey=hotkey).set(median)
            self.response_time_std.labels(hotkey=hotkey).set(std)

    def update_miner_metrics(self, hotkey: str, score: float, stats: dict):
        """Helper method to update all metrics for a given miner"""
        # Only update metrics if there are successful validations
        if len(self.response_times[hotkey]) > 0:
            # Update basic metrics
            self.miner_scores.labels(hotkey=hotkey).set(score)
            self.miner_last_seen.labels(hotkey=hotkey).set(time.time())
            
            # Update score tracking
            self.scores[hotkey].append(score)
            
            # Safely extract statistics from the stats dictionary
            rt_mean, rt_median, rt_std = stats.get("response_time_stats", (0, 0, 0))
            score_mean, score_median, score_std = stats.get("score_stats", (0, 0, 0))
            predictions_match = stats.get("predictions_match", False)
            
            # Update response time statistics with proper value checking
            if rt_mean is not None:
                self.response_time_mean.labels(hotkey=hotkey).set(rt_mean)
            if rt_median is not None:
                self.response_time_median.labels(hotkey=hotkey).set(rt_median)
            if rt_std is not None:
                self.response_time_std.labels(hotkey=hotkey).set(rt_std)
            
            # Update score statistics with proper value checking
            if score_mean is not None:
                self.score_mean.labels(hotkey=hotkey).set(score_mean)
            if score_median is not None:
                self.score_median.labels(hotkey=hotkey).set(score_median)
            if score_std is not None:
                self.score_std.labels(hotkey=hotkey).set(score_std)
                
            # Update prediction accuracy
            self.prediction_accuracy.labels(hotkey=hotkey).set(1.0 if predictions_match else 0.0)
            
            # Update average accuracy
            self.miner_avg_accuracy.labels(hotkey=hotkey).set(
                sum(1 for s in stats.get("predictions_match", []) if s) / len(stats.get("predictions_match", []))
                if stats.get("predictions_match", []) else 0.0
            )

# Create a default instance that can be imported
metrics = ValidatorMetrics(port=6969)