import time
import random
from metrics import ValidatorMetrics
from prometheus_client import REGISTRY, start_http_server

def generate_fake_hotkey():
    """Generate a random hotkey for testing"""
    return f"5{random.randint(1000, 9999)}.miner.test.subnet"

def simulate_miner_activity():
    """Simulate activity from multiple miners with varying performance"""
    # Create metrics instance with a different port for testing
    metrics = ValidatorMetrics(port=8099, start_server=False)
    
    # Start the metrics server
    try:
        start_http_server(8099)
        print("Started Prometheus metrics server on port 8099")
    except OSError as e:
        print(f"Warning: Could not start metrics server: {e}")
        print("Continuing with metric generation anyway...")
    
    # Create a pool of test miners
    test_miners = [generate_fake_hotkey() for _ in range(5)]
    
    print("Starting metric simulation with fake miners:")
    for miner in test_miners:
        print(f"- {miner}")
    
    try:
        while True:
            for hotkey in test_miners:
                # Simulate validation attempt
                success = random.random() > 0.1  # 90% success rate
                duration = random.gauss(2.0, 1.0)  # Mean 2s, stddev 1s
                duration = max(0.1, duration)  # Ensure positive duration
                
                metrics.record_validation_attempt(
                    hotkey=hotkey,
                    success=success,
                    duration=duration
                )
                
                # Simulate varying performance metrics
                if success:
                    accuracy = random.uniform(0.7, 0.99)  # Random accuracy between 70-99%
                    base_score = random.uniform(0.6, 0.95)
                    # Add some temporal correlation to scores
                    score = base_score + random.gauss(0, 0.05)
                    score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                    
                    metrics.update_miner_metrics(
                        hotkey=hotkey,
                        score=score,
                        accuracy=accuracy
                    )
                
                # Update active miners count randomly
                active_count = random.randint(3, len(test_miners))
                metrics.active_miners.set(active_count)
            
            # Sleep for a random interval to simulate real-world variation
            time.sleep(random.uniform(0.5, 2.0))
            
    except KeyboardInterrupt:
        print("\nStopping metric simulation")

if __name__ == "__main__":
    print("Starting fake metrics generation...")
    print("Press Ctrl+C to stop")
    simulate_miner_activity() 