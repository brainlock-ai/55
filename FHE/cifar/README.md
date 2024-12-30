## Important Notes

- Running on personal machines is not recommended
- Use AWS `m6i.metal` instance for optimal performance
- Port 5000 is specified via `--axon.external_port` to broadcast the location of your FHE inference server to the network
- If using AWS or cloud providers, ensure security groups/firewall rules allow traffic on the broadcast port (5000)
