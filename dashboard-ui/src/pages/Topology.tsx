import { useMemo } from "react";
import { Link } from "react-router-dom";
import {
  Wifi,
  WifiOff,
  Server,
  Layers,
  Zap,
  Box,
  ShieldAlert,
  ArrowRightLeft,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import type {
  ClusterSnapshot,
  NodeInfo,
  ReplicaInfo,
  ModelConfigInfo,
  GpuGuardEntry,
} from "../hooks/useCluster";

/* ───────── GPU Chip ───────── */

function GpuChip({
  gpu,
  guardEntry,
  memoryLimit,
}: {
  gpu: {
    memory_used_mb: number;
    memory_total_mb: number;
    state: string;
    index: number;
  };
  guardEntry?: GpuGuardEntry;
  memoryLimit: number;
}) {
  const pct =
    gpu.memory_total_mb > 0 ? gpu.memory_used_mb / gpu.memory_total_mb : 0;
  const gs = guardEntry?.state;

  const bg =
    gs === "MIGRATING"
      ? "bg-purple-500"
      : gs === "MITIGATING"
        ? "bg-orange-500"
        : gs === "WARNED"
          ? "bg-amber-500"
          : pct > memoryLimit
            ? "bg-cs-danger"
            : pct > 0.5
              ? "bg-cs-accent"
              : pct > 0.05
                ? "bg-cs-accent/60"
                : "bg-cs-border2";

  const ring =
    gs === "MIGRATING"
      ? "ring-1 ring-purple-400/50 animate-glow-pulse"
      : gs === "MITIGATING"
        ? "ring-1 ring-orange-400/50 animate-pulse-slow"
        : "";

  return (
    <div className="group relative">
      <div
        className={`w-7 h-7 rounded-md ${bg} ${ring} flex items-center justify-center text-[9px] font-mono font-bold text-white/80 transition-all duration-200 hover:scale-110 hover:brightness-125`}
      >
        {gpu.index}
      </div>
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-cs-card rounded-lg px-3 py-2 text-[10px] whitespace-nowrap z-20 border border-cs-border2 shadow-lg">
        <div className="font-semibold text-cs-text mb-0.5">
          GPU {gpu.index}
        </div>
        <div className="text-cs-muted">
          {gpu.memory_used_mb}MB / {gpu.memory_total_mb}MB (
          {(pct * 100).toFixed(0)}%)
        </div>
        {gs && gs !== "OK" && (
          <div className="mt-1 text-orange-400 font-bold">{gs}</div>
        )}
      </div>
    </div>
  );
}

/* ───────── Status Badge ───────── */

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "READY"
      ? "bg-cs-accent/10 text-cs-accent border border-cs-accent/20"
      : status === "STARTING"
        ? "bg-cs-warn/10 text-cs-warn border border-cs-warn/20 animate-pulse-slow"
        : status === "DRAINING"
          ? "bg-blue-500/10 text-blue-400 border border-blue-500/20"
          : status === "FAILED"
            ? "bg-cs-danger/10 text-cs-danger border border-cs-danger/20"
            : "bg-cs-border text-cs-dim border border-cs-border2";

  return <span className={`badge ${cls}`}>{status}</span>;
}

/* ───────── Node Card ───────── */

function NodeCard({
  node,
  replicas,
  guardEntries,
  memoryLimit,
}: {
  node: NodeInfo;
  replicas: ReplicaInfo[];
  guardEntries: GpuGuardEntry[];
  memoryLimit: number;
}) {
  const nodeReplicas = replicas.filter((r) => r.node_name === node.name);
  const nodeGuard = guardEntries.filter((e) => e.node_name === node.name);
  const hasAlert = nodeGuard.length > 0;
  const online = node.status === "ONLINE";
  const circuitOpen =
    (node.circuit_open_until ?? 0) > 0 &&
    Date.now() / 1000 < (node.circuit_open_until ?? 0);
  const circuitReopenSec = circuitOpen
    ? Math.max(0, Math.round((node.circuit_open_until - Date.now() / 1000)))
    : 0;

  return (
    <div
      className={`card-hover p-4 space-y-3 animate-fade-in ${
        hasAlert ? "border-orange-500/30" : ""
      } ${circuitOpen ? "border-yellow-500/40" : ""} ${
        !online ? "opacity-40" : ""
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className={`w-7 h-7 rounded-lg flex items-center justify-center ${
              circuitOpen
                ? "bg-yellow-500/10 border border-yellow-500/25"
                : online
                ? "bg-cs-accent/8 border border-cs-accent/15"
                : "bg-cs-danger/8 border border-cs-danger/15"
            }`}
          >
            <Server
              className={`w-3.5 h-3.5 ${
                circuitOpen
                  ? "text-yellow-400"
                  : online
                  ? "text-cs-accent"
                  : "text-cs-danger"
              }`}
            />
          </div>
          <div>
            <span className="font-semibold text-[13px]">{node.name}</span>
            <span className="text-[10px] text-cs-dim font-mono ml-2">
              {node.host}
            </span>
          </div>
          {hasAlert && (
            <ShieldAlert className="w-3.5 h-3.5 text-orange-400 animate-pulse" />
          )}
          {circuitOpen && (
            <span
              className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-yellow-500/15 text-yellow-300 border border-yellow-500/25 uppercase tracking-wide"
              title={`Launch circuit open — excluded from placement for ${circuitReopenSec}s. Node is unreachable from control plane.`}
            >
              Circuit Open · {circuitReopenSec}s
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="badge bg-cs-border text-cs-dim border border-cs-border2">
            {node.gpu_type.toUpperCase()}
          </span>
          {online && !circuitOpen ? (
            <Wifi className="w-3.5 h-3.5 text-cs-accent/60" />
          ) : online && circuitOpen ? (
            <Wifi className="w-3.5 h-3.5 text-yellow-400 animate-pulse" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-cs-danger" />
          )}
        </div>
      </div>

      {/* GPU grid */}
      <div>
        <div className="text-[9px] text-cs-dim font-semibold uppercase tracking-widest mb-1.5">
          GPUs
          <span className="ml-2 text-cs-dim/50 normal-case tracking-normal">
            limit {Math.round(memoryLimit * 100)}%
          </span>
        </div>
        <div className="flex gap-1 flex-wrap">
          {(node.gpus || []).map((g) => (
            <GpuChip
              key={g.index}
              gpu={g}
              memoryLimit={memoryLimit}
              guardEntry={nodeGuard.find((e) => e.gpu_index === g.index)}
            />
          ))}
        </div>
      </div>

      {/* Replicas */}
      {nodeReplicas.length > 0 && (
        <div className="space-y-1">
          <div className="text-[9px] text-cs-dim font-semibold uppercase tracking-widest mb-1">
            Replicas
          </div>
          {nodeReplicas.map((r) => (
            <div
              key={r.replica_id}
              className="flex items-center justify-between bg-cs-surface rounded-lg px-3 py-2 border border-cs-border/50"
            >
              <div className="flex items-center gap-2">
                <Layers className="w-3 h-3 text-cs-accent2/70" />
                <span className="text-[12px] font-medium">{r.model}</span>
                <span className="text-[9px] text-cs-dim font-mono">
                  [{r.gpu_ids.join(",")}]
                </span>
              </div>
              <div className="flex items-center gap-2">
                {r.inflight_requests > 0 && (
                  <span className="flex items-center gap-0.5 text-[10px] text-cs-warn font-mono">
                    <Zap className="w-2.5 h-2.5" />
                    {r.inflight_requests}
                  </span>
                )}
                <StatusBadge status={r.status} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ───────── Queue Bar ───────── */

function QueueBar({ model, depth }: { model: string; depth: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-[12px] font-medium w-40 truncate">{model}</span>
      <div className="flex-1 h-1 bg-cs-border rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            depth > 50
              ? "bg-cs-danger"
              : depth > 10
                ? "bg-cs-warn"
                : "bg-cs-accent"
          }`}
          style={{ width: `${Math.min((depth / 200) * 100, 100)}%` }}
        />
      </div>
      <span className="text-[11px] font-mono text-cs-dim w-8 text-right">
        {depth}
      </span>
    </div>
  );
}

/* ───────── Model Deployment Card ───────── */

function ModelDeploymentCard({
  name,
  config,
  replicas,
  inflightTotal,
  queueDepth,
  recentScaleUp,
  hasLaunchFailures,
  modelHref,
}: {
  name: string;
  config: ModelConfigInfo;
  replicas: ReplicaInfo[];
  inflightTotal: number;
  queueDepth: number;
  recentScaleUp?: boolean;
  hasLaunchFailures?: boolean;
  modelHref: string;
}) {
  const readyCount = replicas.filter((r) => r.status === "READY").length;
  const startingCount = replicas.filter((r) => r.status === "STARTING").length;
  const failedCount = replicas.filter((r) => r.status === "FAILED").length;
  const drainingCount = replicas.filter((r) => r.status === "DRAINING").length;
  const totalCount = replicas.length;

  type ModelStatus = "up" | "degraded" | "starting" | "launching" | "failed-retrying" | "scaled-to-zero" | "down";
  let modelStatus: ModelStatus;
  if (readyCount > 0 && failedCount === 0) {
    modelStatus = "up";
  } else if (readyCount > 0 && failedCount > 0) {
    modelStatus = "degraded";
  } else if (startingCount > 0) {
    modelStatus = "starting";
  } else if (totalCount === 0 && recentScaleUp && hasLaunchFailures) {
    modelStatus = "failed-retrying";
  } else if (totalCount === 0 && recentScaleUp) {
    modelStatus = "launching";
  } else if (totalCount === 0) {
    modelStatus = "scaled-to-zero";
  } else {
    modelStatus = "down";
  }

  const statusConfig = {
    up:             { label: "UP",             color: "text-emerald-400", bg: "bg-emerald-400/10 border-emerald-400/20", dot: "bg-emerald-400", pulse: true },
    degraded:       { label: "DEGRADED",       color: "text-amber-400",   bg: "bg-amber-400/10 border-amber-400/20",   dot: "bg-amber-400",   pulse: true },
    starting:       { label: "STARTING",       color: "text-blue-400",    bg: "bg-blue-400/10 border-blue-400/20",     dot: "bg-blue-400",    pulse: true },
    launching:      { label: "LAUNCHING",      color: "text-blue-400",    bg: "bg-blue-400/10 border-blue-400/20",     dot: "bg-blue-400",    pulse: true },
    "failed-retrying": { label: "FAILED – retrying", color: "text-red-400", bg: "bg-red-400/10 border-red-400/20", dot: "bg-red-400", pulse: true },
    "scaled-to-zero": { label: "SCALED TO 0", color: "text-cs-dim",      bg: "bg-cs-border border-cs-border2",        dot: "bg-cs-dim",      pulse: false },
    down:           { label: "DOWN",           color: "text-red-400",     bg: "bg-red-400/10 border-red-400/20",       dot: "bg-red-400",     pulse: true },
  } as const;
  const sc = statusConfig[modelStatus];

  const inner = (
    <div className="card-hover p-5 space-y-4 animate-slide-up h-full relative group/card">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5 min-w-0">
          <Box className="w-4 h-4 text-cs-accent2/70 shrink-0" />
          <span className="text-[14px] font-semibold truncate">{name}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
          <span className="text-[10px] text-cs-accent/80 font-medium opacity-0 group-hover/card:opacity-100 transition-opacity hidden sm:inline">
            Logs & replicas →
          </span>
          <span
            className={`inline-flex items-center gap-1.5 text-[10px] font-semibold px-2.5 py-1 rounded-full border ${sc.bg} ${sc.color}`}
          >
            <span className="relative flex h-2 w-2">
              {sc.pulse && (
                <span
                  className={`absolute inline-flex h-full w-full rounded-full ${sc.dot} opacity-60 animate-ping`}
                />
              )}
              <span className={`relative inline-flex h-2 w-2 rounded-full ${sc.dot}`} />
            </span>
            {sc.label}
          </span>
        </div>
      </div>

      <div className="text-[10px] text-cs-dim font-mono truncate">
        {config.hf_model}
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1.5">
        {[
          { text: `TP=${config.tp}`, accent: true },
          config.node_type_required && {
            text: config.node_type_required.toUpperCase(),
          },
          { text: config.routing_strategy },
          { text: `ctx=${config.engine.max_model_len.toLocaleString()}` },
          { text: `seqs=${config.engine.max_num_seqs}` },
          {
            text: `vram=${Math.round(config.engine.gpu_memory_utilization * 100)}%`,
          },
        ]
          .filter(Boolean)
          .map((tag, i) => (
            <span
              key={i}
              className={`text-[9px] px-2 py-0.5 rounded-full font-mono ${
                (tag as { accent?: boolean }).accent
                  ? "bg-cs-accent/8 text-cs-accent border border-cs-accent/15"
                  : "bg-cs-border text-cs-dim border border-cs-border2"
              }`}
            >
              {(tag as { text: string }).text}
            </span>
          ))}
      </div>

      {/* Replica breakdown */}
      {(totalCount > 0 || modelStatus === "launching" || modelStatus === "failed-retrying") && (
        <div className="flex items-center gap-3 text-[10px] font-mono">
          {modelStatus === "launching" && (
            <span className="flex items-center gap-1 text-blue-400">
              <RefreshCw className="w-3 h-3 animate-spin" /> placing…
            </span>
          )}
          {modelStatus === "failed-retrying" && (
            <span className="flex items-center gap-1 text-red-400">
              <AlertTriangle className="w-3 h-3" /> FAILED – retrying
            </span>
          )}
          {readyCount > 0 && (
            <span className="flex items-center gap-1 text-emerald-400">
              <CheckCircle2 className="w-3 h-3" /> {readyCount} ready
            </span>
          )}
          {startingCount > 0 && (
            <span className="flex items-center gap-1 text-blue-400">
              <RefreshCw className="w-3 h-3 animate-spin" /> {startingCount} starting
            </span>
          )}
          {drainingCount > 0 && (
            <span className="flex items-center gap-1 text-amber-400">
              <ArrowRightLeft className="w-3 h-3" /> {drainingCount} draining
            </span>
          )}
          {failedCount > 0 && (
            <span className="flex items-center gap-1 text-red-400">
              <XCircle className="w-3 h-3" /> {failedCount} failed
            </span>
          )}
        </div>
      )}

      {/* Live metrics */}
      <div className="grid grid-cols-4 gap-3">
        {[
          {
            label: "Replicas",
            value: totalCount,
            sub: `${config.autoscaling.min_replicas}–${config.autoscaling.max_replicas}`,
          },
          {
            label: "Inflight",
            value: inflightTotal,
            sub: `target: ${config.autoscaling.target_inflight}`,
          },
          { label: "Queue", value: queueDepth, sub: "" },
          {
            label: "Zero-Scale",
            value: config.autoscaling.allow_scale_to_zero ? "ON" : "OFF",
            sub: "",
          },
        ].map((s) => (
          <div key={s.label} className="text-center">
            <div className="text-lg font-bold font-mono">{s.value}</div>
            <div className="text-[9px] text-cs-dim uppercase tracking-wider">
              {s.label}
            </div>
            {s.sub && (
              <div className="text-[9px] text-cs-dim/60">{s.sub}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <Link
      to={modelHref}
      className="block rounded-xl no-underline text-inherit focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cs-accent/50"
    >
      {inner}
    </Link>
  );
}

/* ───────── Main Page ───────── */

export default function Topology({
  snapshot,
}: {
  snapshot: ClusterSnapshot | null;
}) {
  if (!snapshot) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-cs-accent border-t-transparent animate-spin" />
          <span className="text-cs-dim text-sm">
            Connecting to cluster...
          </span>
        </div>
      </div>
    );
  }

  const models = useMemo(
    () => Object.keys(snapshot.queue_depths).sort(),
    [snapshot.queue_depths],
  );

  const launchFailuresByModel = useMemo(() => {
    const set = new Set<string>();
    const cutoff = (snapshot.timestamp || 0) - 300;
    for (const f of snapshot.launch_failures ?? []) {
      if ((f.ts ?? 0) >= cutoff) set.add(f.model);
    }
    return set;
  }, [snapshot.launch_failures, snapshot.timestamp]);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* ── KPI Strip ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {[
          { label: "Nodes", value: snapshot.nodes.length, accent: false },
          { label: "Total GPUs", value: snapshot.stats.total_gpus, accent: false },
          { label: "Free GPUs", value: snapshot.stats.free_gpus, accent: true },
          { label: "Replicas", value: snapshot.stats.ready_replicas, accent: true },
          { label: "Inflight", value: snapshot.stats.total_inflight, accent: false },
          {
            label: "Queue",
            value: snapshot.stats.total_queue_depth,
            accent: false,
          },
        ].map((kpi, i) => (
          <div
            key={kpi.label}
            className="kpi-card animate-slide-up"
            style={{ animationDelay: `${i * 50}ms` }}
          >
            <div className="text-[9px] text-cs-dim font-semibold uppercase tracking-[0.15em]">
              {kpi.label}
            </div>
            <div
              className={`text-2xl font-bold font-mono mt-1 ${
                kpi.accent ? "text-cs-accent" : "text-cs-text"
              }`}
            >
              {kpi.value}
            </div>
            {kpi.accent && (
              <div className="absolute -top-10 -right-10 w-24 h-24 bg-cs-accent/[0.03] rounded-full blur-2xl" />
            )}
          </div>
        ))}
      </div>

      {/* ── Queues ── */}
      {models.length > 0 && (
        <div>
          <h2 className="section-title">Model Queues</h2>
          <div className="card p-4 space-y-2.5">
            {models.map((m) => (
              <QueueBar
                key={m}
                model={m}
                depth={snapshot.queue_depths[m] || 0}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Model Deployments ── */}
      {Object.keys(snapshot.model_configs || {}).length > 0 && (
        <div>
          <h2 className="section-title">Model Deployments</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {Object.entries(snapshot.model_configs).map(([name, config]) => {
              const modelReplicas = snapshot.replicas.filter(
                (r) => r.model === name,
              );
              const inf = modelReplicas.reduce(
                (s, r) => s + r.inflight_requests, 0,
              );
              const recentScaleUp = (snapshot.autoscale_events || []).some(
                (e: { model: string; action: string; timestamp: number }) =>
                  e.model === name &&
                  e.action === "SCALE_UP" &&
                  (snapshot.timestamp || 0) - (e.timestamp || 0) < 120,
              );
              return (
                <ModelDeploymentCard
                  key={name}
                  name={name}
                  config={config}
                  replicas={modelReplicas}
                  inflightTotal={inf}
                  queueDepth={snapshot.queue_depths[name] || 0}
                  recentScaleUp={recentScaleUp}
                  hasLaunchFailures={launchFailuresByModel.has(name)}
                  modelHref={`/models/${encodeURIComponent(name)}`}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* ── GPU Memory Guard ── */}
      {(snapshot.gpu_guard?.entries?.length > 0 ||
        snapshot.gpu_guard?.events?.length > 0) && (
        <div>
          <h2 className="section-title flex items-center gap-2">
            <ShieldAlert className="w-3.5 h-3.5 text-orange-400" />
            GPU Memory Guard
            <span className="text-cs-dim/50 normal-case tracking-normal font-normal">
              VRAM limit {Math.round((snapshot.gpu_guard?.memory_limit || 0.92) * 100)}%
              {" · "}compute ≥{" "}
              {Math.round(
                (snapshot.gpu_guard?.compute_sustain_threshold ?? 0.95) * 100,
              )}
              % for {Math.round(snapshot.gpu_guard?.compute_sustain_duration_s ?? 900)}s
              {" · "}window {snapshot.gpu_guard?.mitigation_window_s || 600}s
            </span>
          </h2>
          <div className="space-y-2">
            {(snapshot.gpu_guard?.entries || []).map((e, i) => (
              <div
                key={`${e.node_name}-${e.gpu_index}-${i}`}
                className={`card flex items-center justify-between px-4 py-3 text-xs ${
                  e.state === "MIGRATING"
                    ? "border-purple-500/30 bg-purple-500/[0.03]"
                    : e.state === "MITIGATING"
                      ? "border-orange-500/30 bg-orange-500/[0.03]"
                      : "border-amber-500/30 bg-amber-500/[0.03]"
                }`}
              >
                <div className="flex items-center gap-3">
                  {e.state === "MIGRATING" ? (
                    <ArrowRightLeft className="w-4 h-4 text-purple-400 animate-pulse" />
                  ) : (
                    <ShieldAlert className="w-4 h-4 text-orange-400" />
                  )}
                  <span className="font-semibold">{e.node_name}</span>
                  <span className="font-mono text-cs-dim">
                    GPU {e.gpu_index}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono" title="VRAM used / total">
                    VRAM {Math.round(e.last_utilization * 100)}%
                  </span>
                  {e.last_compute_utilization != null && (
                    <span
                      className="font-mono text-cs-dim"
                      title="GPU compute utilization (nvidia-smi)"
                    >
                      GPU {Math.round(e.last_compute_utilization * 100)}%
                    </span>
                  )}
                  <span
                    className={`badge ${
                      e.state === "MIGRATING"
                        ? "bg-purple-500/10 text-purple-300 border border-purple-500/20"
                        : e.state === "MITIGATING"
                          ? "bg-orange-500/10 text-orange-300 border border-orange-500/20"
                          : "bg-amber-500/10 text-amber-300 border border-amber-500/20"
                    }`}
                  >
                    {e.state}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Nodes Grid ── */}
      <div>
        <h2 className="section-title">Cluster Nodes</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {snapshot.nodes.map((node, i) => (
            <div
              key={node.name}
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <NodeCard
                node={node}
                replicas={snapshot.replicas}
                guardEntries={snapshot.gpu_guard?.entries || []}
                memoryLimit={snapshot.gpu_guard?.memory_limit || 0.92}
              />
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
