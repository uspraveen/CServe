import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Box,
  ChevronRight,
  FileText,
  Layers,
  Loader2,
  RefreshCw,
  Server,
  Stethoscope,
  Terminal,
} from "lucide-react";
import type { ClusterSnapshot, ModelConfigInfo, ReplicaInfo } from "../hooks/useCluster";

type ModelActivity = {
  model: string;
  recent_jobs: Array<{
    job_id: string;
    event: string;
    timestamp: number;
    replica_id: string | null;
    node_name: string | null;
    metadata: Record<string, unknown>;
  }>;
  health_incidents: Array<{
    timestamp: number;
    node_name: string | null;
    replica_id: string | null;
    incident_type: string;
    details: string | null;
  }>;
  remediation: Array<Record<string, unknown>>;
};

type ReplicaDiagnostics = {
  ok: boolean;
  error?: string;
  replica_id: string;
  node_name?: string;
  model?: string;
  http_endpoint?: string;
  status?: string;
  vllm?: {
    output_tail: string;
    exit_code: number | null;
    vllm_metrics: Record<string, number>;
    agent_error: string | null;
    log_path?: string;
    log_offset?: number;
  };
  cserve_job_events?: ModelActivity["recent_jobs"];
  cserve_health_incidents?: ModelActivity["health_incidents"];
  remediation?: Array<Record<string, unknown>>;
};

function RemediationIcon({ action }: { action: string }) {
  if (action === "restart_in_place")
    return <RefreshCw className="w-3.5 h-3.5 text-cs-warn shrink-0" />;
  if (action === "migrate")
    return <Server className="w-3.5 h-3.5 text-purple-400 shrink-0" />;
  if (action === "give_up")
    return <Stethoscope className="w-3.5 h-3.5 text-cs-danger shrink-0" />;
  return <FileText className="w-3.5 h-3.5 text-cs-dim shrink-0" />;
}

function LogPre({
  text,
  empty,
  scrollRef,
}: {
  text: string;
  empty: string;
  scrollRef?: React.RefObject<HTMLPreElement | null>;
}) {
  if (!text?.trim()) {
    return <p className="text-xs text-cs-dim py-4">{empty}</p>;
  }
  return (
    <pre
      ref={scrollRef}
      className="text-[11px] font-mono leading-relaxed text-cs-muted whitespace-pre-wrap break-all max-h-96 overflow-y-auto p-3 log-panel"
    >
      {text}
    </pre>
  );
}

export default function ModelPage({
  snapshot,
}: {
  snapshot: ClusterSnapshot | null;
}) {
  const { modelName: modelParam } = useParams<{ modelName: string }>();
  const modelName = modelParam ? decodeURIComponent(modelParam) : "";

  const [activity, setActivity] = useState<ModelActivity | null>(null);
  const [activityErr, setActivityErr] = useState<string | null>(null);
  const [selectedReplicaId, setSelectedReplicaId] = useState<string | null>(
    null,
  );
  const [diag, setDiag] = useState<ReplicaDiagnostics | null>(null);
  const [diagLoading, setDiagLoading] = useState(false);
  const [diagErr, setDiagErr] = useState<string | null>(null);
  const [logTab, setLogTab] = useState<"vllm" | "cserve_jobs" | "cserve_health">(
    "vllm",
  );
  const [vllmLog, setVllmLog] = useState("");
  const [vllmLogErr, setVllmLogErr] = useState<string | null>(null);
  const [vllmLogPath, setVllmLogPath] = useState("");
  const logOffsetRef = useRef(0);
  const logPreRef = useRef<HTMLPreElement>(null);

  const config: ModelConfigInfo | undefined =
    snapshot?.model_configs?.[modelName];
  const replicas: ReplicaInfo[] =
    snapshot?.replicas.filter((r) => r.model === modelName) ?? [];

  const loadActivity = useCallback(async () => {
    if (!modelName) return;
    setActivityErr(null);
    try {
      const res = await fetch(
        `/dashboard/api/models/${encodeURIComponent(modelName)}/activity`,
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setActivity(await res.json());
    } catch (e) {
      setActivityErr(e instanceof Error ? e.message : "Failed to load");
    }
  }, [modelName]);

  const loadDiagnostics = useCallback(async (rid: string) => {
    setDiagLoading(true);
    setDiagErr(null);
    try {
      const res = await fetch(
        `/dashboard/api/replicas/${encodeURIComponent(rid)}/diagnostics`,
      );
      const data = (await res.json()) as ReplicaDiagnostics;
      if (!data.ok) {
        setDiagErr(data.error || "unknown error");
        setDiag(null);
      } else {
        setDiag(data);
      }
    } catch (e) {
      setDiagErr(e instanceof Error ? e.message : "Failed to load");
      setDiag(null);
    } finally {
      setDiagLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadActivity();
  }, [loadActivity, snapshot?.timestamp]);

  useEffect(() => {
    if (selectedReplicaId) void loadDiagnostics(selectedReplicaId);
    else {
      setDiag(null);
      setDiagErr(null);
    }
  }, [selectedReplicaId, loadDiagnostics]);

  const pollVllmLogs = useCallback(async (rid: string, reset: boolean) => {
    if (reset) {
      logOffsetRef.current = 0;
      setVllmLog("");
      setVllmLogErr(null);
    }
    try {
      const res = await fetch(
        `/dashboard/api/replicas/${encodeURIComponent(rid)}/logs?offset=${logOffsetRef.current}`,
      );
      const data = (await res.json()) as {
        ok: boolean;
        text?: string;
        offset?: number;
        log_path?: string;
        agent_error?: string;
        error?: string;
      };
      if (!data.ok) {
        setVllmLogErr(data.agent_error || data.error || "log fetch failed");
        return;
      }
      setVllmLogErr(null);
      const chunk = data.text || "";
      if (reset) setVllmLog(chunk);
      else if (chunk) setVllmLog((prev) => prev + chunk);
      if (typeof data.offset === "number") logOffsetRef.current = data.offset;
      if (data.log_path) setVllmLogPath(data.log_path);
    } catch (e) {
      setVllmLogErr(e instanceof Error ? e.message : "log poll failed");
    }
  }, []);

  useEffect(() => {
    if (!selectedReplicaId || logTab !== "vllm") return;
    void pollVllmLogs(selectedReplicaId, true);
    const timer = window.setInterval(
      () => void pollVllmLogs(selectedReplicaId, false),
      2000,
    );
    return () => window.clearInterval(timer);
  }, [selectedReplicaId, logTab, pollVllmLogs]);

  useEffect(() => {
    const el = logPreRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [vllmLog]);

  if (!snapshot) {
    return (
      <div className="flex items-center justify-center h-[50vh] text-cs-dim text-sm">
        Connecting…
      </div>
    );
  }

  if (!modelName || !config) {
    return (
      <div className="max-w-lg mx-auto card p-8 text-center">
        <p className="text-cs-muted">Unknown model</p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 mt-4 text-cs-accent text-sm font-medium"
        >
          <ArrowLeft className="w-4 h-4" /> Back to cluster
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-fade-in pb-16">
      <div className="flex flex-wrap items-center gap-4">
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-xs font-medium text-cs-muted hover:text-cs-accent transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Cluster
        </Link>
        <span className="text-cs-border">/</span>
        <span className="text-sm font-semibold text-cs-text flex items-center gap-2">
          <Box className="w-4 h-4 text-cs-accent2/70" />
          {modelName}
        </span>
      </div>

      <header className="card p-6 border-cs-border2/80">
        <div className="flex flex-wrap justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-cs-text">
              {modelName}
            </h1>
            <p className="text-[11px] text-cs-dim font-mono mt-1 truncate max-w-2xl">
              {config.hf_model}
            </p>
          </div>
          <button
            type="button"
            onClick={() => void loadActivity()}
            className="self-start flex items-center gap-2 px-3 py-1.5 rounded-lg border border-cs-border text-[11px] font-medium text-cs-muted hover:text-cs-text hover:border-cs-border2 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh activity
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        <div className="xl:col-span-2 space-y-6">
          <section>
            <h2 className="section-title flex items-center gap-2">
              <Layers className="w-3.5 h-3.5" />
              Replicas
            </h2>
            <p className="text-[11px] text-cs-dim mb-3">
              Select a replica to load vLLM stdout/stderr tail and CServe rows
              for that process.
            </p>
            <ul className="space-y-2">
              {replicas.length === 0 ? (
                <li className="card p-4 text-sm text-cs-muted">
                  No replicas registered for this model right now.
                </li>
              ) : (
                replicas.map((r) => {
                  const sel = r.replica_id === selectedReplicaId;
                  return (
                    <li key={r.replica_id}>
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedReplicaId(r.replica_id);
                          setLogTab("vllm");
                        }}
                        className={`w-full text-left card px-4 py-3 flex items-center justify-between gap-3 transition-all border ${
                          sel
                            ? "border-cs-accent/40 bg-cs-accent/[0.04] shadow-glow-sm"
                            : "border-cs-border hover:border-cs-border2"
                        }`}
                      >
                        <div className="min-w-0">
                          <div className="font-mono text-xs text-cs-text truncate">
                            {r.replica_id}
                          </div>
                          <div className="text-[10px] text-cs-dim mt-0.5">
                            {r.node_name} · GPUs [{r.gpu_ids.join(", ")}]
                          </div>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <span
                            className={`badge ${
                              r.status === "READY"
                                ? "bg-cs-accent/10 text-cs-accent border border-cs-accent/20"
                                : r.status === "FAILED"
                                  ? "bg-cs-danger/10 text-cs-danger border border-cs-danger/20"
                                  : "bg-cs-border text-cs-dim"
                            }`}
                          >
                            {r.status}
                          </span>
                          <ChevronRight
                            className={`w-4 h-4 ${sel ? "text-cs-accent" : "text-cs-dim"}`}
                          />
                        </div>
                      </button>
                    </li>
                  );
                })
              )}
            </ul>
          </section>

          <section>
            <h2 className="section-title flex items-center gap-2">
              <RefreshCw className="w-3.5 h-3.5 text-cs-warn" />
              Auto-remediation (model)
            </h2>
            <div className="card overflow-hidden max-h-52 overflow-y-auto">
              {activityErr ? (
                <p className="p-4 text-xs text-cs-danger">{activityErr}</p>
              ) : !activity ? (
                <p className="p-4 text-xs text-cs-dim">Loading…</p>
              ) : activity.remediation.length === 0 ? (
                <p className="p-4 text-xs text-cs-dim">
                  No recent remediation events for this model.
                </p>
              ) : (
                activity.remediation.map((ev, i) => (
                  <div
                    key={`mrem-${i}`}
                    className="flex items-center gap-3 px-4 py-2.5 table-row"
                  >
                    <RemediationIcon
                      action={String(ev.action || "")}
                    />
                    <span className="font-mono text-cs-dim text-[10px] w-14 shrink-0">
                      {ev.timestamp
                        ? new Date(
                            (ev.timestamp as number) * 1000,
                          ).toLocaleTimeString()
                        : ""}
                    </span>
                    <span className="text-[11px] font-mono text-cs-muted truncate">
                      {(ev.replica_id as string) || "—"}
                    </span>
                    <span className="badge bg-cs-card text-cs-muted border border-cs-border2 shrink-0 text-[9px]">
                      {String(ev.action || "")}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>

          <section>
            <h2 className="section-title">Recent requests (model)</h2>
            <div className="card overflow-hidden max-h-56 overflow-y-auto">
              {activityErr ? null : !activity ? (
                <p className="p-4 text-xs text-cs-dim">Loading…</p>
              ) : activity.recent_jobs.length === 0 ? (
                <p className="p-4 text-xs text-cs-dim">No recent job events.</p>
              ) : (
                activity.recent_jobs.map((job, i) => (
                  <div
                    key={`${job.job_id}-${i}`}
                    className="flex items-center gap-3 px-4 py-2.5 table-row"
                  >
                    <span className="font-mono text-cs-dim text-[10px] w-14 shrink-0 truncate">
                      {job.job_id.slice(0, 8)}
                    </span>
                    <span className="badge bg-cs-border/50 text-cs-text border border-cs-border2 shrink-0">
                      {job.event}
                    </span>
                    <span className="text-[10px] font-mono text-cs-muted truncate">
                      {job.replica_id || "—"}
                    </span>
                    <span className="ml-auto text-cs-dim font-mono text-[10px] shrink-0">
                      {new Date(job.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        <div className="xl:col-span-3">
          <h2 className="section-title flex items-center gap-2">
            <Terminal className="w-3.5 h-3.5" />
            Replica diagnostics
          </h2>
          {!selectedReplicaId ? (
            <div className="card p-8 text-center text-sm text-cs-muted">
              Choose a replica on the left.
            </div>
          ) : (
            <div className="card overflow-hidden">
              <div className="border-b border-cs-border flex flex-wrap gap-1 p-2 bg-cs-surface/50">
                {(
                  [
                    ["vllm", "vLLM (agent)"],
                    ["cserve_jobs", "CServe jobs"],
                    ["cserve_health", "CServe health"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setLogTab(id)}
                    className={`px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors ${
                      logTab === id
                        ? "bg-cs-accent/15 text-cs-accent border border-cs-accent/25"
                        : "text-cs-muted hover:text-cs-text border border-transparent"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="p-4 min-h-[280px]">
                {diagLoading ? (
                  <div className="flex items-center justify-center gap-2 py-16 text-cs-dim text-sm">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Loading…
                  </div>
                ) : diagErr ? (
                  <p className="text-sm text-cs-danger">{diagErr}</p>
                ) : !diag ? (
                  <p className="text-sm text-cs-dim">No data.</p>
                ) : logTab === "vllm" ? (
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center gap-2 text-[10px] text-cs-dim">
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        Live · 2s
                      </span>
                      {vllmLogPath ? (
                        <span className="font-mono truncate max-w-full" title={vllmLogPath}>
                          {vllmLogPath}
                        </span>
                      ) : null}
                      <button
                        type="button"
                        className="ml-auto text-cs-accent hover:underline"
                        onClick={() =>
                          selectedReplicaId &&
                          void pollVllmLogs(selectedReplicaId, true)
                        }
                      >
                        Refresh
                      </button>
                    </div>
                    {vllmLogErr ? (
                      <p className="text-xs text-amber-400/90 border border-amber-500/20 rounded-lg p-3 bg-amber-500/5">
                        {vllmLogErr}
                        <span className="block mt-1 text-cs-dim">
                          Worker node agents must be updated to stream from{" "}
                          <code className="text-[10px]">~/.cserve/logs/vllm-*.log</code>.
                        </span>
                      </p>
                    ) : null}
                    {diag.vllm?.agent_error && !vllmLogErr ? (
                      <p className="text-xs text-amber-400/90 border border-amber-500/20 rounded-lg p-3 bg-amber-500/5">
                        Diagnostics: {diag.vllm.agent_error}
                      </p>
                    ) : null}
                    {diag.vllm &&
                    Object.keys(diag.vllm.vllm_metrics || {}).length > 0 ? (
                      <details className="text-xs">
                        <summary className="cursor-pointer text-cs-dim mb-2">
                          vLLM /metrics (sample)
                        </summary>
                        <pre className="font-mono text-[10px] text-cs-muted max-h-32 overflow-auto p-2 rounded log-panel">
                          {Object.entries(diag.vllm.vllm_metrics)
                            .slice(0, 40)
                            .map(([k, v]) => `${k} ${v}`)
                            .join("\n")}
                        </pre>
                      </details>
                    ) : null}
                    <LogPre
                      scrollRef={logPreRef}
                      text={vllmLog || diag.vllm?.output_tail || ""}
                      empty="Waiting for vLLM log stream from worker node…"
                    />
                  </div>
                ) : logTab === "cserve_jobs" ? (
                  <div className="max-h-80 overflow-y-auto space-y-0">
                    {(diag.cserve_job_events || []).length === 0 ? (
                      <p className="text-xs text-cs-dim py-2">
                        No job_events rows for this replica_id.
                      </p>
                    ) : (
                      (diag.cserve_job_events || []).map((job, i) => (
                        <div
                          key={`dj-${job.job_id}-${i}`}
                          className="flex items-center gap-2 py-2 border-b border-cs-border/40 text-[11px]"
                        >
                          <span className="font-mono text-cs-dim w-14 shrink-0">
                            {job.job_id.slice(0, 8)}
                          </span>
                          <span className="badge shrink-0">{job.event}</span>
                          <span className="text-cs-dim ml-auto font-mono text-[10px]">
                            {new Date(job.timestamp * 1000).toLocaleString()}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                ) : (
                  <div className="max-h-80 overflow-y-auto space-y-2">
                    {(diag.cserve_health_incidents || []).length === 0 ? (
                      <p className="text-xs text-cs-dim">
                        No health_incidents rows for this replica.
                      </p>
                    ) : (
                      (diag.cserve_health_incidents || []).map((h, i) => (
                        <div
                          key={`dh-${i}`}
                          className="text-[11px] border border-cs-border/60 rounded-lg p-3 bg-cs-surface/30"
                        >
                          <div className="flex flex-wrap gap-2 items-center mb-1">
                            <span className="badge bg-cs-warn/10 text-cs-warn border border-cs-warn/20">
                              {h.incident_type}
                            </span>
                            <span className="text-cs-dim font-mono text-[10px]">
                              {new Date(h.timestamp * 1000).toLocaleString()}
                            </span>
                          </div>
                          {h.details ? (
                            <p className="text-cs-muted leading-relaxed">
                              {h.details}
                            </p>
                          ) : null}
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>

              {diag?.remediation && diag.remediation.length > 0 ? (
                <div className="border-t border-cs-border p-4 bg-cs-surface/50">
                  <h3 className="text-[10px] font-semibold text-cs-dim uppercase tracking-widest mb-2">
                    Remediation (this replica)
                  </h3>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {diag.remediation.map((ev, i) => (
                      <div
                        key={`rrem-${i}`}
                        className="flex items-center gap-2 text-[11px]"
                      >
                        <RemediationIcon
                          action={String(ev.action || "")}
                        />
                        <span className="text-cs-muted">
                          {String(ev.action || "")}
                        </span>
                        {ev.failure_class ? (
                          <span className="badge text-[9px]">
                            {String(ev.failure_class)}
                          </span>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
