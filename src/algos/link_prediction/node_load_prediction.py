"""Derive node load forecasts from SCFE's directed link forecasts."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite, sqrt


@dataclass
class _Observation:
    timestamp: datetime
    uploads: dict
    loads: dict
    incident_edges: dict


@dataclass
class _PendingStep:
    origin: _Observation
    previous_timestamp: datetime
    number: int
    node_ids: list
    changes: list


class NodeLoadPrediction:
    def __init__(self, constellation_id):
        self.constellation_id = str(constellation_id)
        self._observation = None
        self._pending = {}

    def observe(self, timestamp, node_flows, topology):
        """Read the actual load exactly once after a micro-cycle routes traffic."""
        uploads = {}
        loads = {}
        for node_id, upload, load in node_flows:
            uploads[str(node_id)] = float(upload)
            loads[str(node_id)] = float(load)
        incident_edges = {node: set() for node in loads}
        for source, neighbors in topology.items():
            for destination in neighbors:
                edge = (str(source), str(destination))
                source_id, destination_id = edge
                if source_id in incident_edges and destination_id in incident_edges:
                    incident_edges[source_id].add(edge)
                    incident_edges[destination_id].add(edge)
        previous = self._observation
        actual = _Observation(timestamp, uploads, loads, incident_edges)
        self._observation = actual

        pending = self._pending.pop(timestamp, None)
        for expired in [target for target in self._pending if target < timestamp]:
            self._pending.pop(expired)
        if pending is None or previous is None or previous.timestamp != pending.previous_timestamp:
            return None
        return self._score_step(pending, previous, actual)

    def predict(self, link_forecast, rated_capacity_gbps):
        """Aggregate full directed SCFE predictions before the UI hot-link filter."""
        origin = self._observation
        if origin is None or not origin.loads:
            return None
        step_seconds = float(link_forecast["TimeStepSeconds"])
        if not isfinite(step_seconds) or step_seconds <= 0:
            return None
        step_duration = timedelta(seconds=step_seconds)
        previous_timestamp = origin.timestamp
        previous_loads = origin.loads
        node_ids = list(origin.loads)
        steps = []
        for number, forecast_step in enumerate(link_forecast["LinksPredTopology"], start=1):
            timestamp = datetime.fromisoformat(forecast_step["Timestamp"])
            if timestamp != previous_timestamp + step_duration:
                break
            predicted_links = defaultdict(dict)
            for source, neighbors in forecast_step["Topology"].items():
                for destination, metrics in neighbors:
                    remaining = metrics.get("Capacity")
                    if remaining is None or not isfinite(remaining) or not 0 <= remaining <= rated_capacity_gbps:
                        continue
                    edge = (str(source), str(destination))
                    # SCFE's full output stores rated capacity minus predicted flow.
                    flow = rated_capacity_gbps - remaining
                    for endpoint in edge:
                        predicted_links[endpoint][edge] = flow
            loads = {}
            changes = []
            for node in node_ids:
                links = predicted_links.get(node, {})
                # Match the active directed edge set and count each edge once.
                if origin.incident_edges[node] != links.keys():
                    loads[node] = None
                else:
                    loads[node] = round(origin.uploads[node] + sum(links.values()), 3)
                current = loads[node]
                previous = previous_loads[node]
                changes.append(round(current - previous, 3) if current is not None and previous is not None else None)
            steps.append({
                "Timestamp": timestamp.isoformat(),
                "PredictedLoadGbps": list(loads.values()),
                "PredictedLoadFluctuationGbps": changes,
            })
            self._pending[timestamp] = _PendingStep(origin, previous_timestamp, number, node_ids, changes)
            previous_timestamp, previous_loads = timestamp, loads
        if not steps:
            return None
        return {
            "ConstellationId": self.constellation_id,
            "OriginTimestamp": origin.timestamp.isoformat(),
            "TimeStepSeconds": step_seconds,
            "GroundUploadMethod": "Persistence",
            "NodeIds": node_ids,
            "ObservedLoadGbps": list(origin.loads.values()),
            "Steps": steps,
        }

    def _score_step(self, pending, previous, actual):
        errors = []
        baseline_errors = []
        skipped_topology = 0
        for node, predicted in zip(pending.node_ids, pending.changes):
            if predicted is None or node not in actual.loads or node not in previous.loads:
                continue
            edges = pending.origin.incident_edges[node]
            if actual.incident_edges[node] != edges or previous.incident_edges[node] != edges:
                skipped_topology += 1
                continue
            measured = actual.loads[node] - previous.loads[node]
            errors.append(abs(predicted - measured))
            baseline_errors.append(abs(measured))
        samples = len(errors)
        mae = sum(errors) / samples if samples else None
        baseline_mae = sum(baseline_errors) / samples if samples else None
        return {
            "ConstellationId": self.constellation_id,
            "ForecastOriginTimestamp": pending.origin.timestamp.isoformat(),
            "Timestamp": actual.timestamp.isoformat(),
            "Step": pending.number,
            "Samples": samples,
            "SkippedTopologyNodes": skipped_topology,
            "MAEGbps": mae,
            "RMSEGbps": sqrt(sum(error * error for error in errors) / samples) if samples else None,
            "ZeroChangeBaselineMAEGbps": baseline_mae,
            "MAEImprovementVsZeroChangePct": (1 - mae / baseline_mae) * 100 if baseline_mae else None,
        }
