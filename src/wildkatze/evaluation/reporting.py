from typing import Dict, List
from datetime import datetime
import json

class EvaluationReport:
    """Generates evaluation and compliance reports."""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.utcnow().isoformat()
        
    def add_benchmark(self, name: str, results: Dict):
        """Add benchmark results to report."""
        self.results[name] = results
        
    def generate_summary(self) -> Dict:
        """Generate summary report."""
        summary = {
            "timestamp": self.timestamp,
            "benchmarks_run": len(self.results),
            "results": self.results,
            "overall_status": self._compute_status()
        }
        return summary
    
    def _compute_status(self) -> str:
        # Check if all benchmarks passed
        for name, result in self.results.items():
            if isinstance(result, dict) and result.get("passed") is False:
                return "FAILED"
        return "PASSED"
    
    def to_markdown(self) -> str:
        """Export report as Markdown."""
        lines = [
            f"# Evaluation Report",
            f"**Generated:** {self.timestamp}",
            f"**Status:** {self._compute_status()}",
            "",
            "## Benchmark Results",
            ""
        ]
        
        for name, result in self.results.items():
            lines.append(f"### {name}")
            if isinstance(result, dict):
                for k, v in result.items():
                    lines.append(f"- **{k}:** {v}")
            else:
                lines.append(f"- {result}")
            lines.append("")
            
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps(self.generate_summary(), indent=2)
