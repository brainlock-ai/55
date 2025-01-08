## Important Notes

- Running on personal machines is not recommended
- Use AWS `m6i.metal` instance for optimal performance
- Port 5000 is specified via `--axon.external_port` to broadcast the location of your FHE inference server to the network
- If using AWS or cloud providers, ensure security groups/firewall rules allow traffic on the broadcast port (5000)
- PostgreSQL runs on port 5432 by default, configurable via `POSTGRES_PORT` in `.env`

## Reserved Ports

The following ports are already in use by the system and should not be configured for other purposes:
- Port 5000: FHE inference server broadcast
- Port 6969: Prometheus metrics
- Port 6970: Reserved for system use
- Port 6971: Reserved for system use
- Port 8091: Stake verification service
