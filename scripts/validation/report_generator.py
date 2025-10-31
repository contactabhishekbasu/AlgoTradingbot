"""
HTML and JSON Report Generator for Validation Results
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from .base import ValidationPhase


class ReportGenerator:
    """Generate validation reports in multiple formats"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path('validation_reports')
        self.output_dir.mkdir(exist_ok=True)

    def generate_all_reports(self, phases: List[ValidationPhase], metadata: Dict[str, Any] = None):
        """Generate all report formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate JSON report
        json_path = self.generate_json_report(phases, metadata, timestamp)

        # Generate HTML report
        html_path = self.generate_html_report(phases, metadata, timestamp)

        # Generate text summary
        text_path = self.generate_text_summary(phases, metadata, timestamp)

        return {
            'json': json_path,
            'html': html_path,
            'text': text_path
        }

    def generate_json_report(self, phases: List[ValidationPhase], metadata: Dict[str, Any] = None, timestamp: str = None) -> Path:
        """Generate JSON report"""
        timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

        report = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'summary': self._generate_summary(phases),
            'phases': [phase.to_dict() for phase in phases]
        }

        output_path = self.output_dir / f'validation_report_{timestamp}.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return output_path

    def generate_text_summary(self, phases: List[ValidationPhase], metadata: Dict[str, Any] = None, timestamp: str = None) -> Path:
        """Generate plain text summary"""
        timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

        lines = []
        lines.append("=" * 80)
        lines.append("AlgoTradingbot Validation Report")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        if metadata:
            lines.append("Metadata:")
            for key, value in metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        summary = self._generate_summary(phases)
        lines.append("Overall Summary:")
        lines.append(f"  Total Phases: {summary['total_phases']}")
        lines.append(f"  Phases Passed: {summary['phases_passed']}")
        lines.append(f"  Total Checks: {summary['total_checks']}")
        lines.append(f"  Checks Passed: {summary['checks_passed']}")
        lines.append(f"  Pass Rate: {summary['overall_pass_rate']:.1f}%")
        lines.append(f"  Total Duration: {summary['total_duration_seconds']:.1f}s")
        lines.append(f"  Status: {'✅ PASS' if summary['all_passed'] else '❌ FAIL'}")
        lines.append("")

        for phase in phases:
            lines.append("-" * 80)
            lines.append(f"Phase {phase.phase_number}: {phase.phase_name}")
            lines.append(f"  Status: {'✅ PASS' if phase.passed else '❌ FAIL'}")
            lines.append(f"  Checks: {phase.passed_count}/{phase.total_count} passed ({phase.pass_rate:.1f}%)")
            lines.append(f"  Duration: {phase.duration_seconds:.1f}s")
            lines.append("")

            for result in phase.results:
                status_symbol = result.status.value
                lines.append(f"  {status_symbol} {result.name}")
                if result.message:
                    lines.append(f"      {result.message}")
                lines.append(f"      Duration: {result.duration_ms:.1f}ms")

            lines.append("")

        lines.append("=" * 80)

        output_path = self.output_dir / f'validation_summary_{timestamp}.txt'
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        return output_path

    def generate_html_report(self, phases: List[ValidationPhase], metadata: Dict[str, Any] = None, timestamp: str = None) -> Path:
        """Generate HTML report"""
        timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

        summary = self._generate_summary(phases)

        html = self._generate_html_template(summary, phases, metadata)

        output_path = self.output_dir / f'validation_report_{timestamp}.html'
        with open(output_path, 'w') as f:
            f.write(html)

        return output_path

    def _generate_summary(self, phases: List[ValidationPhase]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_phases = len(phases)
        phases_passed = sum(1 for p in phases if p.passed)
        total_checks = sum(p.total_count for p in phases)
        checks_passed = sum(p.passed_count for p in phases)
        total_duration = sum(p.duration_seconds for p in phases)

        return {
            'total_phases': total_phases,
            'phases_passed': phases_passed,
            'total_checks': total_checks,
            'checks_passed': checks_passed,
            'overall_pass_rate': (checks_passed / total_checks * 100) if total_checks > 0 else 0,
            'total_duration_seconds': total_duration,
            'all_passed': phases_passed == total_phases
        }

    def _generate_html_template(self, summary: Dict, phases: List[ValidationPhase], metadata: Dict = None) -> str:
        """Generate HTML report template"""
        status_color = '#28a745' if summary['all_passed'] else '#dc3545'
        status_text = 'PASS' if summary['all_passed'] else 'FAIL'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlgoTradingbot Validation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px;
                      box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 40px; }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .subtitle {{ color: #7f8c8d; margin-bottom: 30px; }}
        .summary {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 30px;
                    border-left: 4px solid {status_color}; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                         gap: 20px; margin-top: 20px; }}
        .summary-item {{ text-align: center; }}
        .summary-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .summary-label {{ color: #7f8c8d; margin-top: 5px; font-size: 14px; }}
        .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px;
                         background: {status_color}; color: white; font-weight: bold;
                         font-size: 18px; margin-bottom: 20px; }}
        .phase {{ margin-bottom: 30px; border: 1px solid #e1e8ed; border-radius: 8px; overflow: hidden; }}
        .phase-header {{ background: #f8f9fa; padding: 15px 20px; border-bottom: 1px solid #e1e8ed;
                         display: flex; justify-content: space-between; align-items: center; }}
        .phase-title {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
        .phase-stats {{ color: #7f8c8d; font-size: 14px; }}
        .phase-body {{ padding: 20px; }}
        .check-item {{ padding: 12px; margin-bottom: 8px; border-radius: 6px; background: #f8f9fa;
                       display: flex; justify-content: space-between; align-items: center; }}
        .check-item.pass {{ border-left: 3px solid #28a745; }}
        .check-item.fail {{ border-left: 3px solid #dc3545; }}
        .check-item.warn {{ border-left: 3px solid #ffc107; }}
        .check-name {{ font-weight: 500; color: #2c3e50; }}
        .check-message {{ color: #7f8c8d; font-size: 14px; margin-top: 4px; }}
        .check-duration {{ color: #7f8c8d; font-size: 12px; }}
        .status-icon {{ font-size: 20px; margin-right: 10px; }}
        .metadata {{ background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; }}
        .metadata-item {{ color: #555; margin: 5px 0; }}
        .progress-bar {{ height: 8px; background: #e1e8ed; border-radius: 4px; overflow: hidden;
                         margin-top: 10px; }}
        .progress-fill {{ height: 100%; background: {status_color}; transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AlgoTradingbot Validation Report</h1>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

        <div class="status-badge">{status_text}</div>

        <div class="summary">
            <h2>Overall Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-value">{summary['phases_passed']}/{summary['total_phases']}</div>
                    <div class="summary-label">Phases Passed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{summary['checks_passed']}/{summary['total_checks']}</div>
                    <div class="summary-label">Checks Passed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{summary['overall_pass_rate']:.1f}%</div>
                    <div class="summary-label">Pass Rate</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{summary['total_duration_seconds']:.1f}s</div>
                    <div class="summary-label">Total Duration</div>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {summary['overall_pass_rate']:.1f}%"></div>
            </div>
        </div>
"""

        if metadata:
            html += """
        <div class="metadata">
            <h3>Metadata</h3>
"""
            for key, value in metadata.items():
                html += f"""            <div class="metadata-item"><strong>{key}:</strong> {value}</div>\n"""
            html += """        </div>\n"""

        # Add phases
        for phase in phases:
            phase_status = 'pass' if phase.passed else 'fail'
            html += f"""
        <div class="phase">
            <div class="phase-header">
                <div class="phase-title">Phase {phase.phase_number}: {phase.phase_name}</div>
                <div class="phase-stats">
                    {phase.passed_count}/{phase.total_count} passed ({phase.pass_rate:.1f}%) • {phase.duration_seconds:.1f}s
                </div>
            </div>
            <div class="phase-body">
"""
            for result in phase.results:
                check_status = 'pass' if result.passed else 'fail'
                status_icon = '✅' if result.passed else '❌'

                html += f"""
                <div class="check-item {check_status}">
                    <div>
                        <div class="check-name">
                            <span class="status-icon">{status_icon}</span>
                            {result.name}
                        </div>
                        {f'<div class="check-message">{result.message}</div>' if result.message else ''}
                    </div>
                    <div class="check-duration">{result.duration_ms:.1f}ms</div>
                </div>
"""

            html += """
            </div>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html
