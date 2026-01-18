import pytest

def test_docker_compose_valid():
    """Verify docker-compose.yml syntax."""
    import yaml
    with open("docker/docker-compose.yml", "r") as f:
        config = yaml.safe_load(f)
    
    assert "services" in config
    assert "api" in config["services"]

def test_kubernetes_manifests():
    """Verify Kubernetes manifests exist."""
    from pathlib import Path
    
    k8s_dir = Path("kubernetes")
    required_files = ["deployment.yaml", "service.yaml", "ingress.yaml", "configmap.yaml"]
    
    for f in required_files:
        assert (k8s_dir / f).exists(), f"Missing {f}"
