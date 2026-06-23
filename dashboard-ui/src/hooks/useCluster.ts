import { useEffect, useRef, useState, useCallback } from "react";

export interface GpuInfo {
  index: number;
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization_pct: number;
  temperature_c: number;
  state: string;
  allocated_replica_id: string | null;
}

export interface NodeInfo {
  name: string;
  host: string;
  status: string;
  gpu_type: string;
  gpus: GpuInfo[];
  // Circuit breaker state — populated by the control plane registry
  consecutive_launch_failures: number;
  circuit_open_until: number; // unix timestamp; 0 = circuit closed
}

export interface ReplicaInfo {
  replica_id: string;
  model: string;
  node_name: string;
  gpu_ids: number[];
  status: string;
  inflight_requests: number;
  http_endpoint: string;
}

export interface AutoscaleEvent {
  timestamp: number;
  model: string;
  action: string;
  from_replicas: number;
  to_replicas: number;
  reasons: string[];
}

export interface JobEvent {
  job_id: string;
  event: string;
  timestamp: number;
  replica_id: string | null;
  node_name: string | null;
  metadata: Record<string, unknown>;
}

export interface ModelConfigInfo {
  served_model_name: string;
  hf_model: string;
  tp: number;
  node_type_required: string | null;
  node_types_allowed: string[];
  routing_strategy: string;
  capabilities: string[];
  engine: {
    max_model_len: number;
    max_num_seqs: number;
    gpu_memory_utilization: number;
    dtype: string;
  };
  autoscaling: {
    min_replicas: number;
    max_replicas: number;
    allow_scale_to_zero: boolean;
    idle_timeout_s: number;
    target_inflight: number;
  };
}

export interface GpuGuardEntry {
  node_name: string;
  gpu_index: number;
  replica_id: string;
  state: string;
  consecutive_breaches: number;
  last_utilization: number;
  peak_utilization: number;
  mitigation_started_at: number;
  last_compute_utilization?: number;
  compute_sustain_started_at?: number;
}

export interface GpuGuardEvent {
  timestamp: number;
  node_name: string;
  gpu_index: number;
  replica_id: string;
  model: string;
  old_state: string;
  new_state: string;
  utilization: number;
  action: string;
  details: string;
}

export interface GpuGuardState {
  memory_limit: number;
  compute_sustain_threshold?: number;
  compute_sustain_duration_s?: number;
  mitigation_window_s: number;
  entries: GpuGuardEntry[];
  events: GpuGuardEvent[];
}

/** Live GPU compute pressure (≥ sustain threshold); includes OK while timer runs. */
export interface GpuComputeNotification {
  node_name: string;
  gpu_index: number;
  replica_id: string;
  guard_state: string;
  compute_util_frac: number;
  threshold_frac: number;
  sustain_duration_s: number;
  compute_sustain_started_at: number;
  sustain_elapsed_s: number;
  sustain_progress: number;
}

export interface RemediationEntry {
  timestamp: number;
  action: string;
  failure_class: string;
  replica_id: string;
  model: string;
  node_name: string;
  attempt?: number;
  total_retries?: number;
  from_node?: string;
  reason?: string;
  output_snippet?: string;
  gpus?: number[];
}

export interface LaunchFailure {
  model: string;
  node: string;
  error: string;
  ts: number;
}

export interface ClusterSnapshot {
  timestamp: number;
  nodes: NodeInfo[];
  replicas: ReplicaInfo[];
  models: string[];
  model_configs: Record<string, ModelConfigInfo>;
  queue_depths: Record<string, number>;
  autoscale_events: AutoscaleEvent[];
  health_incidents: Array<Record<string, unknown>>;
  recent_jobs: JobEvent[];
  gpu_guard: GpuGuardState;
  gpu_compute_notifications?: GpuComputeNotification[];
  remediation_log: RemediationEntry[];
  launch_failures?: LaunchFailure[];
  stats: {
    total_gpus: number;
    free_gpus: number;
    total_replicas: number;
    ready_replicas: number;
    total_inflight: number;
    total_queue_depth: number;
  };
}

export function useCluster() {
  const [snapshot, setSnapshot] = useState<ClusterSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/dashboard/ws`;
    const ws = new WebSocket(url);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "ping") return;
        setSnapshot(data);
      } catch {}
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { snapshot, connected };
}
