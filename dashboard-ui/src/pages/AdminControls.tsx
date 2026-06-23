import { useEffect, useState, useCallback } from "react";
import {
  Settings,
  Shield,
  Cpu,
  Save,
  AlertTriangle,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Check,
  Loader2,
  Server,
  Plus,
  Trash2,
  Wifi,
  WifiOff,
  ScanLine,
  Terminal,
  KeyRound,
  GitBranch,
  X,
  Power,
} from "lucide-react";

interface SafetyConfig {
  gpu_memory_limit: number;
  gpu_warn_threshold: number;
  gpu_danger_threshold: number;
  gpu_compute_sustain_threshold: number;
  gpu_compute_sustain_duration_s: number;
  guard_mitigation_window_s: number;
  guard_check_interval_s: number;
}

function ControlPlaneLogsPanel() {
  const [expanded, setExpanded] = useState(false);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/dashboard/api/control-plane/logs?lines=400");
      const data = (await res.json()) as { ok: boolean; text?: string; error?: string };
      if (!data.ok) {
        setErr(data.error || "failed");
        return;
      }
      setErr(null);
      setText(data.text || "");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "fetch failed");
    }
  }, []);

  useEffect(() => {
    if (!expanded) return;
    void load();
    const t = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(t);
  }, [expanded, load]);

  return (
    <div className="card overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((p) => !p)}
        className="w-full flex items-center gap-2 px-5 py-4 border-b border-cs-border text-left hover:bg-cs-surface/40"
      >
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-cs-dim" />
        ) : (
          <ChevronRight className="w-4 h-4 text-cs-dim" />
        )}
        <Terminal className="w-4 h-4 text-cs-accent" />
        <span className="text-[13px] font-semibold">Control plane logs (cosmos-9)</span>
        <span className="text-[10px] text-emerald-400 ml-2">journalctl · 3s</span>
      </button>
      {expanded && (
        <div className="p-4">
          {err ? (
            <p className="text-xs text-red-400">{err}</p>
          ) : (
            <pre className="text-[10px] font-mono text-cs-muted whitespace-pre-wrap break-all max-h-80 overflow-y-auto p-3 rounded-lg log-panel">
              {text || "Loading…"}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

const DEFAULT_SAFETY: SafetyConfig = {
  gpu_memory_limit: 0.79,
  gpu_warn_threshold: 0.70,
  gpu_danger_threshold: 0.79,
  gpu_compute_sustain_threshold: 0.99,
  gpu_compute_sustain_duration_s: 900,
  guard_mitigation_window_s: 600,
  guard_check_interval_s: 20,
};

interface ModelEngineConfig {
  max_num_seqs: number;
  gpu_memory_utilization: number;
  enable_chunked_prefill: boolean;
  enable_prefix_caching: boolean;
  max_model_len: number;
}

interface ModelAutoscaleConfig {
  min_replicas: number;
  max_replicas: number;
  allow_scale_to_zero: boolean;
  target_inflight: number;
  idle_timeout_s: number;
  upscale_cooldown_s: number;
  downscale_cooldown_s: number;
  max_queue_depth: number;
  replica_startup_timeout_s: number;
}

interface ModelAdminConfig {
  engine: ModelEngineConfig;
  autoscaling: ModelAutoscaleConfig;
}

interface AdminConfig {
  safety: SafetyConfig;
  models: Record<string, ModelAdminConfig>;
}

interface SaveResult {
  ok: boolean;
  changes: string[];
  requires_restart: string[];
  warning: string | null;
  /** Keys written to SQLite ``ui_runtime_tuning`` on the control plane. */
  persisted_to_sqlite?: string[];
  /** YAML paths updated on save (cluster.yaml + models.yaml). */
  persisted_to_yaml?: string[];
}

interface SshConfigData {
  username: string;
  key_path: string;
  password: string | null;
  has_password: boolean;
  port: number;
  timeout_s: number;
  cserve_src: string;
  python_path: string;
  pip_path: string;
}

interface NodeInfo {
  name: string;
  host: string;
  status: string;
  gpu_type: string;
  gpu_count: number;
  agent_endpoint: string;
  last_heartbeat: number;
  replica_count: number;
  schedulable: boolean;
  labels: Record<string, string>;
}

interface GpuProbeInfo {
  index: number;
  name: string;
  memory_total_mb: number;
  utilization_pct: number;
}

type WizardStep = "idle" | "form" | "probing" | "probed" | "deploying" | "done" | "error";

function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}) {
  return (
    <div className="space-y-1">
      <label className="text-[11px] text-cs-muted font-medium">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step || 1}
        className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-cs-accent/50 focus:ring-1 focus:ring-cs-accent/20 outline-none transition-all"
      />
      {hint && <p className="text-[9px] text-cs-dim">{hint}</p>}
    </div>
  );
}

function ToggleInput({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}) {
  return (
    <div className="flex items-center justify-between py-1">
      <div>
        <span className="text-[11px] text-cs-muted font-medium">{label}</span>
        {hint && (
          <p className="text-[9px] text-cs-dim mt-0.5">{hint}</p>
        )}
      </div>
      <button
        onClick={() => onChange(!value)}
        className={`relative w-10 h-5 rounded-full transition-colors duration-200 ${
          value
            ? "bg-cs-accent"
            : "bg-cs-border2"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform duration-200 ${
            value ? "translate-x-5" : "translate-x-0"
          }`}
        />
      </button>
    </div>
  );
}

const ADMIN_KEY_STORAGE = "cserve.admin.apiKey";

export default function AdminControls() {
  // ── Core config state ────────────────────────────────────────────────────
  const [config, setConfig] = useState<AdminConfig | null>(null);
  const [original, setOriginal] = useState<AdminConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<SaveResult | null>(null);
  const [restartingModel, setRestartingModel] = useState<string | null>(null);
  const [restartResult, setRestartResult] = useState<string | null>(null);
  const [purgeLoading, setPurgeLoading] = useState(false);
  const [purgeResult, setPurgeResult] = useState<string | null>(null);
  const [clusterStopLoading, setClusterStopLoading] = useState(false);
  const [clusterStopResult, setClusterStopResult] = useState<string | null>(null);
  const [clusterResumeLoading, setClusterResumeLoading] = useState(false);
  const [clusterResumeResult, setClusterResumeResult] = useState<string | null>(null);
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  // ── Persistent admin API key ─────────────────────────────────────────────
  const [adminKey, setAdminKey] = useState<string>(() =>
    window.sessionStorage.getItem(ADMIN_KEY_STORAGE) || ""
  );
  const persistKey = (key: string) => {
    setAdminKey(key);
    if (key) window.sessionStorage.setItem(ADMIN_KEY_STORAGE, key);
    else window.sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  };

  // ── SSH config state ─────────────────────────────────────────────────────
  const [sshConfig, setSshConfig] = useState<SshConfigData | null>(null);
  const [sshExpanded, setSshExpanded] = useState(false);
  const [savingSsh, setSavingSsh] = useState(false);
  const [sshSaveMsg, setSshSaveMsg] = useState<{ok: boolean; msg: string} | null>(null);

  // ── Cluster topology state ───────────────────────────────────────────────
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [nodesExpanded, setNodesExpanded] = useState(true);
  const [removingNode, setRemovingNode] = useState<string | null>(null);
  const [togglingSchedulable, setTogglingSchedulable] = useState<string | null>(null);

  // ── Add Server wizard state ──────────────────────────────────────────────
  const [wizardStep, setWizardStep] = useState<WizardStep>("idle");
  const [wizardName, setWizardName] = useState("");
  const [wizardHost, setWizardHost] = useState("");
  const [wizardGpuType, setWizardGpuType] = useState("");
  const [wizardSyncCode, setWizardSyncCode] = useState(true);
  const [probeResult, setProbeResult] = useState<{ gpus: GpuProbeInfo[]; hostname: string; os_info: string; error: string | null } | null>(null);
  const [selectedGpus, setSelectedGpus] = useState<Set<number>>(new Set());
  const [wizardLog, setWizardLog] = useState<string[]>([]);
  const [wizardError, setWizardError] = useState<string | null>(null);

  const resetWizard = () => {
    setWizardStep("idle");
    setWizardName(""); setWizardHost(""); setWizardGpuType("");
    setWizardSyncCode(true); setProbeResult(null);
    setSelectedGpus(new Set()); setWizardLog([]); setWizardError(null);
  };

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch("/dashboard/api/admin_config");
      const data = await res.json();
      const merged = {
        ...data,
        safety: { ...DEFAULT_SAFETY, ...data.safety },
      };
      setConfig(merged);
      setOriginal(JSON.parse(JSON.stringify(merged)));
    } catch {
      /* retry silently */
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchSshConfig = useCallback(async (key: string) => {
    if (!key) return;
    try {
      const res = await fetch("/admin/ssh_config", {
        headers: { Authorization: `Bearer ${key}` },
      });
      if (res.ok) setSshConfig(await res.json());
    } catch { /* silent */ }
  }, []);

  const fetchNodes = useCallback(async (key: string) => {
    if (!key) return;
    try {
      const res = await fetch("/admin/nodes", {
        headers: { Authorization: `Bearer ${key}` },
      });
      if (res.ok) {
        const data = await res.json();
        setNodes(data.nodes || []);
      }
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  useEffect(() => {
    if (adminKey) {
      fetchSshConfig(adminKey);
      fetchNodes(adminKey);
    }
  }, [adminKey, fetchSshConfig, fetchNodes]);

  const hasChanges = config && original
    ? JSON.stringify(config) !== JSON.stringify(original)
    : false;

  const hasEngineChanges = (modelName: string) => {
    if (!config || !original) return false;
    const c = config.models[modelName]?.engine;
    const o = original.models[modelName]?.engine;
    return c && o && JSON.stringify(c) !== JSON.stringify(o);
  };

  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    setSaveResult(null);
    setRestartResult(null);

    const apiKey = adminKey || prompt(
      "Enter your admin API key (csk_...).\n\nThis is required to save configuration changes."
    );
    if (!apiKey) {
      setSaving(false);
      return;
    }
    if (apiKey !== adminKey) persistKey(apiKey);

    try {
      const res = await fetch("/admin/config", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      if (res.ok) {
        setSaveResult({
          ok: true,
          changes: data.changes ?? [],
          requires_restart: data.requires_restart ?? [],
          warning: data.warning ?? null,
          persisted_to_sqlite: data.persisted_to_sqlite,
          persisted_to_yaml: data.persisted_to_yaml,
        });
        setOriginal(JSON.parse(JSON.stringify(config)));
      } else {
        setSaveResult({
          ok: false,
          changes: [],
          requires_restart: [],
          warning: data.error?.message || "Save failed",
        });
      }
    } catch (e) {
      setSaveResult({
        ok: false,
        changes: [],
        requires_restart: [],
        warning: `Network error: ${e}`,
      });
    } finally {
      setSaving(false);
    }
  };

  const handleRestart = async (modelName: string) => {
    const apiKey = adminKey || prompt(
      `Rolling restart for "${modelName}".\n\n` +
      "This will drain each replica (completing inflight requests) " +
      "then restart with new settings. No requests will be lost.\n\n" +
      "Enter your admin API key to proceed:"
    );
    if (!apiKey) return;
    if (apiKey !== adminKey) persistKey(apiKey);

    setRestartingModel(modelName);
    setRestartResult(null);

    try {
      const res = await fetch(`/admin/restart/${modelName}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      const data = await res.json();
      setRestartResult(
        res.ok
          ? data.message
          : data.error?.message || "Restart failed"
      );
    } catch (e) {
      setRestartResult(`Network error: ${e}`);
    } finally {
      setRestartingModel(null);
    }
  };

  const handlePurgeQueues = async () => {
    const apiKey = adminKey || prompt(
      "Purge ALL CServe Redis job queues?\n\n"
      + "This deletes pending overflow-queue work (streams + priority sets) "
      + "and clears callback/cancelled keys. In-flight gateway waits may "
      + "never complete — restart replicas afterward.\n\n"
      + "Enter your admin API key:",
    );
    if (!apiKey) return;
    if (apiKey !== adminKey) persistKey(apiKey);

    setPurgeLoading(true);
    setPurgeResult(null);
    try {
      const res = await fetch("/admin/queues/purge", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (res.ok) {
        const cleared = (data.models_cleared as string[] | undefined)?.join(", ");
        setPurgeResult(
          `Purged Redis queues (${data.stream_keys_deleted ?? 0} streams, `
          + `${data.priority_keys_deleted ?? 0} priority keys, `
          + `${data.callback_keys_deleted ?? 0} cb keys). `
          + `Models touched: ${cleared || "none"}. `
          + (data.next_steps as string),
        );
      } else {
        setPurgeResult(data.error?.message || "Purge failed");
      }
    } catch (e) {
      setPurgeResult(`Network error: ${e}`);
    } finally {
      setPurgeLoading(false);
    }
  };

  const handleClusterStop = async () => {
    const apiKey = adminKey || prompt(
      "STOP ENTIRE CLUSTER?\n\n"
      + "• Stops all replicas (drain when READY unless you force)\n"
      + "• Purges Redis job queues by default\n"
      + "• Pauses autoscaling until you click Resume\n"
      + "• Sweeps CServe-managed GPUs only (cuda_devices per node)\n\n"
      + "Enter admin API key:",
    );
    if (!apiKey) return;
    if (apiKey !== adminKey) persistKey(apiKey);

    setClusterStopLoading(true);
    setClusterStopResult(null);
    try {
      const res = await fetch("/admin/cluster/stop", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          purge_queues: true,
          wait_after_stop_s: 8,
          second_wait_s: 2,
          cleanup_orphans: true,
          force: false,
          resume_autoscale_after: false,
          verify_gpus: true,
          retry_sweep_on_verify_fail: true,
          // Kill every nvidia-smi process on cuda_devices (not only vLLM/python).
          vllm_only_gpu_sweep: false,
          // Still call the agent if the registry says OFFLINE (stale state).
          sweep_offline_nodes: true,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        const gv = data.gpu_verification as
          | { all_passed?: boolean; nodes?: Record<string, { issues?: string[] }> }
          | undefined;
        const gpuLine = gv
          ? gv.all_passed
            ? "GPU check: managed GPUs look idle (VRAM / allocation)."
            : `GPU check: FAILED — ${Object.entries(gv.nodes ?? {})
                .flatMap(([n, v]) => (v.issues?.length ? [`${n}: ${v.issues.join("; ")}`] : []))
                .join(" | ") || "see gpu_verification in response"}`
          : "";
        const parts = [
          `Stopped ${(data.replicas as unknown[])?.length ?? 0} replica(s).`,
          data.hint as string,
          data.queues
            ? `Queues: ${(data.queues as { models_cleared?: string[] }).models_cleared?.join(", ") || "cleared"}.`
            : "",
          gpuLine,
          typeof data.control_plane_process === "string"
            ? `Control plane: ${data.control_plane_process}`
            : "",
        ].filter(Boolean);
        setClusterStopResult(parts.join(" "));
      } else {
        setClusterStopResult(data.error?.message || "Cluster stop failed");
      }
    } catch (e) {
      setClusterStopResult(`Network error: ${e}`);
    } finally {
      setClusterStopLoading(false);
    }
  };

  const handleClusterResume = async () => {
    const apiKey = adminKey || prompt("Resume autoscaling?\n\nEnter admin API key:");
    if (!apiKey) return;
    if (apiKey !== adminKey) persistKey(apiKey);
    setClusterResumeLoading(true);
    setClusterResumeResult(null);
    try {
      const res = await fetch("/admin/cluster/resume", {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      const data = await res.json();
      setClusterResumeResult(
        res.ok
          ? "Autoscaling resumed — replicas may launch per min/max policy."
          : data.error?.message || "Resume failed",
      );
    } catch (e) {
      setClusterResumeResult(`Network error: ${e}`);
    } finally {
      setClusterResumeLoading(false);
    }
  };

  const toggleModel = (name: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  const handleSaveSsh = async () => {
    if (!sshConfig || !adminKey) return;
    setSavingSsh(true);
    setSshSaveMsg(null);
    try {
      const res = await fetch("/admin/ssh_config", {
        method: "PUT",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${adminKey}` },
        body: JSON.stringify(sshConfig),
      });
      const data = await res.json();
      setSshSaveMsg({ ok: res.ok, msg: res.ok ? "SSH configuration saved." : (data.error?.message || "Save failed") });
    } catch (e) {
      setSshSaveMsg({ ok: false, msg: `Network error: ${e}` });
    } finally {
      setSavingSsh(false);
    }
  };

  const handleToggleSchedulable = async (nodeName: string, schedulable: boolean) => {
    if (!adminKey) return;
    setTogglingSchedulable(nodeName);
    try {
      const res = await fetch(`/admin/nodes/${nodeName}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${adminKey}`,
        },
        body: JSON.stringify({ schedulable }),
      });
      const data = await res.json();
      if (res.ok) {
        setNodes((prev) =>
          prev.map((n) => (n.name === nodeName ? { ...n, schedulable } : n)),
        );
      } else {
        alert(data.error?.message || "Failed to update schedulable flag.");
      }
    } catch (e) {
      alert(`Network error: ${e}`);
    } finally {
      setTogglingSchedulable(null);
    }
  };

  const handleRemoveNode = async (nodeName: string) => {
    if (!adminKey) return;
    const confirmed = window.confirm(
      `Remove node "${nodeName}" from the cluster?\n\n` +
      "This will stop the node agent and deregister the node. " +
      "Any active replicas on this node must be drained first."
    );
    if (!confirmed) return;
    setRemovingNode(nodeName);
    try {
      const res = await fetch(`/admin/nodes/${nodeName}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${adminKey}` },
        body: JSON.stringify({ force: false }),
      });
      const data = await res.json();
      if (res.ok) {
        setNodes((prev) => prev.filter((n) => n.name !== nodeName));
      } else {
        alert(data.error?.message || data.error || "Failed to remove node.");
      }
    } catch (e) {
      alert(`Network error: ${e}`);
    } finally {
      setRemovingNode(null);
    }
  };

  const handleProbeNode = async () => {
    if (!wizardHost || !adminKey) return;
    setWizardStep("probing");
    setProbeResult(null);
    try {
      const res = await fetch("/admin/nodes/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${adminKey}` },
        body: JSON.stringify({ host: wizardHost }),
      });
      const data = await res.json();
      setProbeResult(data);
      if (data.error) {
        setWizardStep("form");
      } else {
        // Auto-detect GPU type from probe result
        if (data.gpus?.length > 0 && !wizardGpuType) {
          const firstName = (data.gpus[0].name || "").toLowerCase();
          if (firstName.includes("a40")) setWizardGpuType("a40");
          else if (firstName.includes("a100")) setWizardGpuType("a100");
          else if (firstName.includes("h100")) setWizardGpuType("h100");
          else if (firstName.includes("l40")) setWizardGpuType("l40");
          else if (firstName.includes("v100")) setWizardGpuType("v100");
          else if (firstName.includes("3090")) setWizardGpuType("rtx3090");
          else if (firstName.includes("4090")) setWizardGpuType("rtx4090");
        }
        // Pre-select all GPUs
        setSelectedGpus(new Set(data.gpus.map((g: GpuProbeInfo) => g.index)));
        setWizardStep("probed");
      }
    } catch (e) {
      setProbeResult({ gpus: [], hostname: "", os_info: "", error: `Network error: ${e}` });
      setWizardStep("form");
    }
  };

  const handleDeployNode = async () => {
    if (!wizardName || !wizardHost || selectedGpus.size === 0 || !adminKey) return;
    setWizardStep("deploying");
    setWizardLog([]);
    setWizardError(null);
    const cudaDevices = [...selectedGpus].sort((a, b) => a - b).join(",");
    try {
      const res = await fetch("/admin/nodes", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${adminKey}` },
        body: JSON.stringify({
          name: wizardName,
          host: wizardHost,
          cuda_devices: cudaDevices,
          gpu_type: wizardGpuType,
          sync_code: wizardSyncCode,
        }),
      });
      const data = await res.json();
      setWizardLog(data.log || []);
      if (res.ok && data.ok) {
        setWizardStep("done");
        fetchNodes(adminKey); // Refresh node list
      } else {
        setWizardError(data.error || "Deployment failed");
        setWizardStep("error");
      }
    } catch (e) {
      setWizardError(`Network error: ${e}`);
      setWizardStep("error");
    }
  };

  const updateSafety = (field: keyof SafetyConfig, value: number) => {
    if (!config) return;
    setConfig({
      ...config,
      safety: { ...config.safety, [field]: value },
    });
  };

  const updateEngine = (
    model: string,
    field: keyof ModelEngineConfig,
    value: number | boolean,
  ) => {
    if (!config) return;
    setConfig({
      ...config,
      models: {
        ...config.models,
        [model]: {
          ...config.models[model],
          engine: { ...config.models[model].engine, [field]: value },
        },
      },
    });
  };

  const updateAutoscale = (
    model: string,
    field: keyof ModelAutoscaleConfig,
    value: number | boolean,
  ) => {
    if (!config) return;
    setConfig({
      ...config,
      models: {
        ...config.models,
        [model]: {
          ...config.models[model],
          autoscaling: {
            ...config.models[model].autoscaling,
            [field]: value,
          },
        },
      },
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-6 h-6 text-cs-accent animate-spin" />
          <span className="text-cs-dim text-sm">Loading configuration...</span>
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <span className="text-cs-danger text-sm">Failed to load config</span>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cs-accent/8 border border-cs-accent/15 flex items-center justify-center">
            <Settings className="w-4 h-4 text-cs-accent" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">Admin Controls</h1>
            <p className="text-[11px] text-cs-dim">
              Saving writes the same fields to{" "}
              <span className="font-mono text-cs-muted">models.yaml</span> (per-model
              engine + autoscaling) and{" "}
              <span className="font-mono text-cs-muted">cluster.yaml</span>{" "}
              (<span className="font-mono">safety.*</span>) on the control plane.
              Autoscaling applies immediately; engine changes need a rolling restart.
            </p>
          </div>
        </div>
        <button
          disabled={!hasChanges || saving}
          onClick={handleSave}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            hasChanges
              ? "bg-cs-accent text-black hover:bg-cs-accent/90 shadow-glow-sm"
              : "bg-cs-border text-cs-dim cursor-not-allowed"
          }`}
        >
          {saving ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Save className="w-3.5 h-3.5" />
          )}
          Save Changes
        </button>
      </div>

      {/* Admin API Key */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-4">
          <KeyRound className="w-4 h-4 text-cs-accent" />
          <h2 className="text-[13px] font-semibold">Admin API Key</h2>
          <span className="text-[10px] text-cs-dim ml-1">stored in session — never persisted to disk</span>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="password"
            value={adminKey}
            onChange={(e) => persistKey(e.target.value)}
            placeholder="csk_admin_..."
            className="flex-1 bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-cs-accent/50 outline-none transition-all"
          />
          {adminKey && (
            <span className="flex items-center gap-1.5 text-[11px] text-emerald-400 font-medium">
              <Check className="w-3.5 h-3.5" /> Key set
            </span>
          )}
        </div>
        <p className="text-[10px] text-cs-dim mt-2">
          Enter your admin key once here and all operations on this page will use it automatically.
          Required for SSH config, node management, config saves, and rolling restarts.
        </p>
      </div>

      {/* Save Result */}
      {saveResult && (
        <div
          className={`card p-4 border ${
            saveResult.ok
              ? "border-cs-accent/30 bg-cs-accent/[0.03]"
              : "border-cs-danger/30 bg-cs-danger/[0.03]"
          }`}
        >
          <div className="flex items-start gap-3">
            {saveResult.ok ? (
              <Check className="w-4 h-4 text-cs-accent mt-0.5 shrink-0" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-cs-danger mt-0.5 shrink-0" />
            )}
            <div className="space-y-2 flex-1">
              {saveResult.changes.length > 0 && (
                <div>
                  <p className="text-[11px] text-cs-muted font-semibold mb-1">
                    Changes Applied:
                  </p>
                  {saveResult.changes.map((c, i) => (
                    <p key={i} className="text-[11px] font-mono text-cs-text">
                      {c}
                    </p>
                  ))}
                </div>
              )}
              {saveResult.warning && (
                <div className="flex items-start gap-2 p-3 rounded-lg bg-cs-warn/[0.05] border border-cs-warn/20">
                  <AlertTriangle className="w-3.5 h-3.5 text-cs-warn mt-0.5 shrink-0" />
                  <p className="text-[11px] text-cs-warn">{saveResult.warning}</p>
                </div>
              )}
              {saveResult.requires_restart.length > 0 && (
                <div className="space-y-1">
                  <p className="text-[11px] text-cs-warn font-semibold">
                    Requires Restart:
                  </p>
                  {saveResult.requires_restart.map((r, i) => (
                    <p key={i} className="text-[11px] font-mono text-cs-warn/80">
                      {r}
                    </p>
                  ))}
                </div>
              )}
              {saveResult.persisted_to_sqlite &&
                saveResult.persisted_to_sqlite.length > 0 && (
                <div className="space-y-1 border-t border-cs-border/50 pt-2">
                  <p className="text-[11px] text-cs-muted font-semibold">
                    Persisted to SQLite (survives restart)
                  </p>
                  {saveResult.persisted_to_sqlite.map((p, i) => (
                    <p key={i} className="text-[10px] font-mono text-cs-dim break-all">
                      {p}
                    </p>
                  ))}
                </div>
              )}
              {saveResult.persisted_to_yaml &&
                saveResult.persisted_to_yaml.length > 0 && (
                <div className="space-y-1 border-t border-cs-border/50 pt-2">
                  <p className="text-[11px] text-cs-muted font-semibold">
                    Mirrored to YAML (Git-friendly)
                  </p>
                  <p className="text-[9px] text-cs-dim">
                    Hand-edited YAML still wins on restart if newer than SQLite.
                    Force reload:{" "}
                    <span className="font-mono">POST /admin/config/sync-from-yaml</span>
                  </p>
                  {saveResult.persisted_to_yaml.map((p, i) => (
                    <p key={i} className="text-[10px] font-mono text-cs-dim break-all">
                      {p}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Restart Result */}
      {restartResult && (
        <div className="card p-4 border border-cs-accent/30 bg-cs-accent/[0.03]">
          <div className="flex items-start gap-3">
            <RefreshCw className="w-4 h-4 text-cs-accent mt-0.5 shrink-0" />
            <p className="text-[11px] text-cs-text">{restartResult}</p>
          </div>
        </div>
      )}

      {/* Queue purge — fresh start */}
      <div className="card p-5 border border-cs-danger/20 bg-cs-danger/[0.02]">
        <div className="flex items-center gap-2 mb-2">
          <Trash2 className="w-4 h-4 text-cs-danger" />
          <h2 className="text-[13px] font-semibold text-cs-text">Redis job queues</h2>
        </div>
        <p className="text-[11px] text-cs-muted leading-relaxed mb-3">
          Empty all CServe overflow queues in Redis (pending jobs that were buffered
          when replicas were saturated). This does{" "}
          <span className="text-cs-warn font-medium">not</span> clear vLLM&apos;s
          internal batching queues — after purging, run a{" "}
          <span className="font-mono text-cs-dim">rolling restart</span> per model
          (below) or restart workers so vLLM starts clean, then restart the control
          plane if you want a fully aligned cold start.
        </p>
        <p className="text-[11px] text-cs-dim leading-relaxed mb-4 border-l-2 border-cs-border pl-3">
          <span className="text-cs-muted font-medium">SQLite (events.db)</span> stores
          historical job events, autoscale audit, and health incidents — it is{" "}
          <span className="text-cs-warn font-medium">not</span> a spillover buffer when
          Redis fills. There is no automatic &quot;Redis → SQLite&quot; queue overflow;
          live pending work stays in Redis until purged or consumed.
        </p>
        <button
          type="button"
          disabled={purgeLoading}
          onClick={() => void handlePurgeQueues()}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-cs-danger/40 text-cs-danger text-[12px] font-medium hover:bg-cs-danger/10 transition-colors disabled:opacity-50"
        >
          {purgeLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Trash2 className="w-4 h-4" />
          )}
          Purge all queues (Redis)
        </button>
        {purgeResult && (
          <p className="mt-3 text-[11px] text-cs-text leading-relaxed border-t border-cs-border/50 pt-3">
            {purgeResult}
          </p>
        )}
      </div>

      {/* Full cluster stop */}
      <div className="card p-5 border border-red-500/25 bg-red-500/[0.03]">
        <div className="flex items-center gap-2 mb-2">
          <Power className="w-4 h-4 text-red-400" />
          <h2 className="text-[13px] font-semibold text-cs-text">Stop entire cluster</h2>
        </div>
        <p className="text-[11px] text-cs-muted leading-relaxed mb-3">
          Stops every replica, purges Redis queues, waits, then{" "}
          <span className="text-cs-warn font-medium">SIGKILLs every GPU process</span>{" "}
          nvidia-smi reports on the indices in{" "}
          <span className="font-mono text-cs-dim">cuda_devices</span> per node
          (dedicated inference cluster — not just vLLM). Sweeps are attempted even
          if a node is marked OFFLINE in the registry (stale health). Autoscaling
          is <span className="text-cs-warn font-medium">paused</span> afterward.
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            disabled={clusterStopLoading}
            onClick={() => void handleClusterStop()}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/15 border border-red-500/35 text-red-300 text-[12px] font-medium hover:bg-red-500/25 transition-colors disabled:opacity-50"
          >
            {clusterStopLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Power className="w-4 h-4" />
            )}
            Stop cluster + GPU sweep
          </button>
          <button
            type="button"
            disabled={clusterResumeLoading}
            onClick={() => void handleClusterResume()}
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-cs-border text-[12px] font-medium text-cs-muted hover:text-cs-text transition-colors disabled:opacity-50"
          >
            {clusterResumeLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            Resume autoscaling
          </button>
        </div>
        {clusterStopResult && (
          <p className="mt-3 text-[11px] text-cs-text leading-relaxed border-t border-cs-border/50 pt-3">
            {clusterStopResult}
          </p>
        )}
        {clusterResumeResult && (
          <p className="mt-2 text-[11px] text-cs-accent leading-relaxed">
            {clusterResumeResult}
          </p>
        )}
      </div>

      {/* ── SSH Configuration ── */}
      <div className="card overflow-hidden">
        <button
          type="button"
          onClick={() => setSshExpanded((p) => !p)}
          className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-cs-hover transition-colors"
        >
          <div className="flex items-center gap-2">
            {sshExpanded ? <ChevronDown className="w-4 h-4 text-cs-dim" /> : <ChevronRight className="w-4 h-4 text-cs-dim" />}
            <Terminal className="w-4 h-4 text-violet-400" />
            <span className="text-[13px] font-semibold">SSH Configuration</span>
            <span className="text-[10px] text-cs-dim">credentials used for node agent deployment</span>
          </div>
          {sshConfig && (
            <span className="text-[10px] font-mono text-cs-dim">
              {sshConfig.username}@host:{sshConfig.port} ·{" "}
              {sshConfig.has_password ? "password auth" : "key auth"} · {sshConfig.python_path}
            </span>
          )}
        </button>
        {sshExpanded && (
          <div className="border-t border-cs-border px-5 py-5 space-y-4">
            {!adminKey && (
              <p className="text-[11px] text-cs-warn flex items-center gap-2">
                <AlertTriangle className="w-3.5 h-3.5" /> Enter your admin API key above to load and save SSH config.
              </p>
            )}
            {sshConfig ? (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[11px] text-cs-muted font-medium">SSH Username</label>
                    <input
                      value={sshConfig.username}
                      onChange={(e) => setSshConfig({ ...sshConfig, username: e.target.value })}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                    <p className="text-[9px] text-cs-dim">Username for SSH login on all worker nodes</p>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[11px] text-cs-muted font-medium">SSH Key Path</label>
                    <input
                      value={sshConfig.key_path}
                      onChange={(e) => setSshConfig({ ...sshConfig, key_path: e.target.value })}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                    <p className="text-[9px] text-cs-dim">
                      Private key on the control plane host. Common: <code className="text-cs-accent">~/.ssh/id_ed25519</code> or <code className="text-cs-accent">~/.ssh/id_rsa</code>.
                      Run <code className="text-cs-accent">ls ~/.ssh/*.pub</code> to see available keys.
                    </p>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <label className="text-[11px] text-cs-muted font-medium">SSH Password</label>
                      {sshConfig.has_password && !sshConfig.password && (
                        <span className="text-[9px] text-amber-400 border border-amber-400/20 bg-amber-400/10 px-1.5 py-0.5 rounded">password set</span>
                      )}
                    </div>
                    <input
                      type="password"
                      value={sshConfig.password ?? ""}
                      onChange={(e) => setSshConfig({ ...sshConfig, password: e.target.value || null })}
                      placeholder={sshConfig.has_password ? "••••••••  (leave blank to keep current)" : "optional — leave blank to use key auth"}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                    <p className="text-[9px] text-cs-dim">
                      Used when key-based auth is not available. If both are set, password takes precedence.
                      Send an empty value to clear the stored password.
                    </p>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[11px] text-cs-muted font-medium">SSH Port</label>
                    <input
                      type="number"
                      value={sshConfig.port}
                      onChange={(e) => setSshConfig({ ...sshConfig, port: parseInt(e.target.value) || 22 })}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[11px] text-cs-muted font-medium">Connection Timeout (s)</label>
                    <input
                      type="number"
                      value={sshConfig.timeout_s}
                      onChange={(e) => setSshConfig({ ...sshConfig, timeout_s: parseFloat(e.target.value) || 30 })}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[11px] text-cs-muted font-medium">Python Path (on nodes)</label>
                    <input
                      value={sshConfig.python_path}
                      onChange={(e) => setSshConfig({ ...sshConfig, python_path: e.target.value })}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[11px] text-cs-muted font-medium">Pip Path (on nodes)</label>
                    <input
                      value={sshConfig.pip_path}
                      onChange={(e) => setSshConfig({ ...sshConfig, pip_path: e.target.value })}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                  </div>
                  <div className="col-span-2 space-y-1">
                    <label className="text-[11px] text-cs-muted font-medium">CServe Source Dir (on control plane)</label>
                    <input
                      value={sshConfig.cserve_src}
                      onChange={(e) => setSshConfig({ ...sshConfig, cserve_src: e.target.value })}
                      className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-violet-400/50 outline-none transition-all"
                    />
                    <p className="text-[9px] text-cs-dim">Local path rsync'd to each new node during deployment</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleSaveSsh}
                    disabled={savingSsh || !adminKey}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-500/10 border border-violet-500/20 text-violet-300 text-sm font-medium transition-colors hover:bg-violet-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {savingSsh ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                    Save SSH Config
                  </button>
                  {sshSaveMsg && (
                    <span className={`text-[11px] font-medium ${sshSaveMsg.ok ? "text-emerald-400" : "text-red-400"}`}>
                      {sshSaveMsg.ok ? <Check className="inline w-3.5 h-3.5 mr-1" /> : <AlertTriangle className="inline w-3.5 h-3.5 mr-1" />}
                      {sshSaveMsg.msg}
                    </span>
                  )}
                </div>
              </>
            ) : (
              adminKey && <p className="text-[11px] text-cs-dim">Loading SSH config…</p>
            )}
          </div>
        )}
      </div>

      {/* ── Cluster Topology ── */}
      <div className="card overflow-hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-cs-border">
          <button
            type="button"
            onClick={() => setNodesExpanded((p) => !p)}
            className="flex items-center gap-2 text-left"
          >
            {nodesExpanded ? <ChevronDown className="w-4 h-4 text-cs-dim" /> : <ChevronRight className="w-4 h-4 text-cs-dim" />}
            <Server className="w-4 h-4 text-sky-400" />
            <span className="text-[13px] font-semibold">Cluster Topology</span>
            <span className="badge bg-cs-border text-cs-dim ml-1">{nodes.length} nodes</span>
          </button>
          {wizardStep === "idle" && (
            <button
              type="button"
              onClick={() => { if (!adminKey) { alert("Enter your admin API key first."); return; } setWizardStep("form"); }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cs-accent/10 border border-cs-accent/20 text-cs-accent text-[11px] font-semibold transition-colors hover:bg-cs-accent/20"
            >
              <Plus className="w-3.5 h-3.5" /> Add Server
            </button>
          )}
        </div>

        {nodesExpanded && (
          <div className="px-5 py-4 space-y-4">
            {/* Node List */}
            {nodes.length === 0 ? (
              <p className="text-[11px] text-cs-dim py-2">{adminKey ? "No nodes loaded — ensure the admin key is valid." : "Enter your admin API key above to view nodes."}</p>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2">
                {nodes.map((node) => {
                  const isOnline = node.status === "ONLINE";
                  const isDegraded = node.status === "DEGRADED";
                  return (
                    <div key={node.name} className="rounded-xl border border-cs-border bg-cs-surface/60 p-4 space-y-2.5">
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            {isOnline ? (
                              <Wifi className="w-3.5 h-3.5 text-emerald-400" />
                            ) : isDegraded ? (
                              <Wifi className="w-3.5 h-3.5 text-amber-400" />
                            ) : (
                              <WifiOff className="w-3.5 h-3.5 text-red-400" />
                            )}
                            <span className="text-[13px] font-semibold text-cs-text">{node.name}</span>
                            <span className={`text-[9px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded-full border ${
                              isOnline ? "text-emerald-300 border-emerald-400/20 bg-emerald-400/10"
                              : isDegraded ? "text-amber-300 border-amber-400/20 bg-amber-400/10"
                              : "text-red-300 border-red-400/20 bg-red-400/10"
                            }`}>{node.status}</span>
                          </div>
                          <p className="text-[10px] font-mono text-cs-dim mt-0.5">{node.host}</p>
                        </div>
                        <button
                          type="button"
                          disabled={removingNode === node.name}
                          onClick={() => handleRemoveNode(node.name)}
                          className="p-1.5 rounded-lg text-cs-dim transition-colors hover:text-red-400 hover:bg-red-400/10 disabled:opacity-50"
                          title="Remove node"
                        >
                          {removingNode === node.name ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                        </button>
                      </div>
                      <div className="flex items-center justify-between rounded-lg border border-cs-border bg-cs-card px-3 py-2">
                        <div>
                          <p className="text-[10px] font-semibold text-cs-text">Schedulable</p>
                          <p className="text-[9px] text-cs-dim">
                            {node.schedulable !== false
                              ? "Accepts new replica placements"
                              : "Excluded from placement (maintenance)"}
                          </p>
                        </div>
                        <button
                          type="button"
                          disabled={togglingSchedulable === node.name}
                          onClick={() =>
                            handleToggleSchedulable(node.name, node.schedulable === false)
                          }
                          className={`relative w-10 h-5 rounded-full transition-colors ${
                            node.schedulable !== false ? "bg-emerald-500/80" : "bg-cs-border2"
                          } disabled:opacity-50`}
                          title={node.schedulable !== false ? "Disable scheduling" : "Enable scheduling"}
                        >
                          <span
                            className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                              node.schedulable !== false ? "translate-x-5" : ""
                            }`}
                          />
                        </button>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-[10px]">
                        <div className="rounded-lg bg-cs-card border border-cs-border px-2.5 py-1.5 text-center">
                          <div className="font-semibold text-cs-text">{node.gpu_count}</div>
                          <div className="text-cs-dim">GPUs</div>
                        </div>
                        <div className="rounded-lg bg-cs-card border border-cs-border px-2.5 py-1.5 text-center">
                          <div className="font-semibold text-cs-text uppercase">{node.gpu_type || "—"}</div>
                          <div className="text-cs-dim">Type</div>
                        </div>
                        <div className="rounded-lg bg-cs-card border border-cs-border px-2.5 py-1.5 text-center">
                          <div className="font-semibold text-cs-text">{node.replica_count}</div>
                          <div className="text-cs-dim">Replicas</div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* ── Add Server Wizard ── */}
            {wizardStep !== "idle" && (
              <div className="rounded-xl border border-cs-accent/20 bg-cs-accent/[0.03] p-5 space-y-5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <GitBranch className="w-4 h-4 text-cs-accent" />
                    <span className="text-[13px] font-semibold text-cs-accent">Add New Server</span>
                  </div>
                  <button type="button" onClick={resetWizard} className="p-1 rounded text-cs-dim hover:text-cs-text">
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Step indicator */}
                <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider">
                  {(["form", "probing", "probed", "deploying", "done", "error"] as WizardStep[]).map((step, i) => {
                    const labels: Record<string, string> = { form: "1. Info", probing: "2. Probe", probed: "3. GPUs", deploying: "4. Deploy", done: "✓ Done", error: "✗ Error" };
                    const stepOrder = { form: 0, probing: 1, probed: 2, deploying: 3, done: 4, error: 4 };
                    const currentOrder = stepOrder[wizardStep] ?? 0;
                    const thisOrder = i;
                    const isActive = step === wizardStep;
                    const isDone = thisOrder < currentOrder;
                    return (
                      <span key={step} className={`px-2 py-1 rounded ${
                        isActive ? "bg-cs-accent/20 text-cs-accent border border-cs-accent/30"
                        : isDone ? "text-emerald-400"
                        : "text-cs-dim"
                      }`}>{labels[step]}</span>
                    );
                  })}
                </div>

                {/* Step 1: Server info */}
                {(wizardStep === "form" || wizardStep === "probing") && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1">
                        <label className="text-[11px] text-cs-muted font-medium">Node Name</label>
                        <input
                          value={wizardName}
                          onChange={(e) => setWizardName(e.target.value)}
                          placeholder="e.g. cosmos-15"
                          className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-cs-accent/50 outline-none"
                        />
                        <p className="text-[9px] text-cs-dim">Unique identifier in the cluster</p>
                      </div>
                      <div className="space-y-1">
                        <label className="text-[11px] text-cs-muted font-medium">Host / IP</label>
                        <input
                          value={wizardHost}
                          onChange={(e) => setWizardHost(e.target.value)}
                          placeholder="e.g. 192.168.1.20 or host.example.com"
                          className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-cs-accent/50 outline-none"
                        />
                        <p className="text-[9px] text-cs-dim">Reachable from the control plane</p>
                      </div>
                    </div>
                    {probeResult?.error && (
                      <div className="flex items-start gap-2 rounded-lg border border-red-400/20 bg-red-400/10 px-3 py-2.5 text-[11px] text-red-300">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <span>{probeResult.error}</span>
                      </div>
                    )}
                    <button
                      type="button"
                      disabled={!wizardName || !wizardHost || wizardStep === "probing"}
                      onClick={handleProbeNode}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-sky-500/10 border border-sky-500/20 text-sky-300 text-sm font-medium transition-colors hover:bg-sky-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {wizardStep === "probing" ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <ScanLine className="w-3.5 h-3.5" />
                      )}
                      {wizardStep === "probing" ? "Probing GPUs…" : "Detect GPUs"}
                    </button>
                    <p className="text-[10px] text-cs-dim">
                      SSH into the server using the credentials above and run nvidia-smi to discover available GPUs.
                    </p>
                  </div>
                )}

                {/* Step 3: GPU Selection */}
                {wizardStep === "probed" && probeResult && (
                  <div className="space-y-4">
                    {probeResult.hostname && (
                      <div className="flex items-center gap-4 rounded-lg bg-cs-card border border-cs-border px-4 py-2.5 text-[11px]">
                        <span className="text-cs-muted">Host:</span>
                        <span className="font-mono text-cs-text">{probeResult.hostname}</span>
                        {probeResult.os_info && (
                          <>
                            <span className="text-cs-dim">·</span>
                            <span className="text-cs-dim">{probeResult.os_info}</span>
                          </>
                        )}
                      </div>
                    )}
                    <div>
                      <p className="text-[12px] font-semibold mb-2 flex items-center gap-1.5">
                        <Cpu className="w-3.5 h-3.5 text-sky-400" />
                        Select GPUs to include ({selectedGpus.size}/{probeResult.gpus.length} selected)
                      </p>
                      <div className="grid gap-2 sm:grid-cols-2">
                        {probeResult.gpus.map((gpu) => {
                          const isSelected = selectedGpus.has(gpu.index);
                          return (
                            <button
                              key={gpu.index}
                              type="button"
                              onClick={() => {
                                setSelectedGpus((prev) => {
                                  const next = new Set(prev);
                                  isSelected ? next.delete(gpu.index) : next.add(gpu.index);
                                  return next;
                                });
                              }}
                              className={`flex items-center gap-3 rounded-xl border p-3 text-left transition-all ${
                                isSelected
                                  ? "border-cs-accent/30 bg-cs-accent/10 shadow-glow-sm"
                                  : "border-cs-border bg-cs-surface/60 hover:border-cs-border2"
                              }`}
                            >
                              <div className={`flex h-8 w-8 items-center justify-center rounded-lg text-[11px] font-bold shrink-0 ${
                                isSelected ? "bg-cs-accent/20 text-cs-accent" : "bg-cs-card text-cs-dim"
                              }`}>
                                {gpu.index}
                              </div>
                              <div className="min-w-0">
                                <div className="text-[11px] font-semibold text-cs-text truncate">{gpu.name}</div>
                                <div className="text-[9px] font-mono text-cs-dim">
                                  {gpu.memory_total_mb > 0 ? `${Math.round(gpu.memory_total_mb / 1024)} GB` : "—"}
                                  {gpu.utilization_pct > 0 && ` · ${gpu.utilization_pct.toFixed(0)}% util`}
                                </div>
                              </div>
                              {isSelected && <Check className="w-3.5 h-3.5 text-cs-accent ml-auto shrink-0" />}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1">
                        <label className="text-[11px] text-cs-muted font-medium">GPU Type Label</label>
                        <input
                          value={wizardGpuType}
                          onChange={(e) => setWizardGpuType(e.target.value)}
                          placeholder="e.g. a40, h100, l40"
                          className="w-full bg-cs-surface border border-cs-border2 rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:border-cs-accent/50 outline-none"
                        />
                        <p className="text-[9px] text-cs-dim">Auto-detected — edit if needed</p>
                      </div>
                      <div className="flex items-center justify-between rounded-lg border border-cs-border bg-cs-surface/50 px-3 py-2.5 self-start mt-5">
                        <div>
                          <span className="text-[11px] text-cs-muted font-medium">Sync Code</span>
                          <p className="text-[9px] text-cs-dim">rsync CServe source before deploying</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => setWizardSyncCode((p) => !p)}
                          className={`relative w-10 h-5 rounded-full transition-colors ${wizardSyncCode ? "bg-cs-accent" : "bg-cs-border2"}`}
                        >
                          <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${wizardSyncCode ? "translate-x-5" : ""}`} />
                        </button>
                      </div>
                    </div>
                    <button
                      type="button"
                      disabled={selectedGpus.size === 0}
                      onClick={handleDeployNode}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cs-accent/10 border border-cs-accent/20 text-cs-accent text-sm font-semibold transition-colors hover:bg-cs-accent/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Server className="w-3.5 h-3.5" />
                      Deploy & Register ({selectedGpus.size} GPU{selectedGpus.size !== 1 ? "s" : ""})
                    </button>
                  </div>
                )}

                {/* Step 4: Deploying */}
                {wizardStep === "deploying" && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-[12px] text-sky-300">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Deploying node agent — this may take 1–2 minutes…
                    </div>
                  </div>
                )}

                {/* Step 5: Done / Error */}
                {(wizardStep === "done" || wizardStep === "error") && (
                  <div className="space-y-3">
                    <div className={`flex items-start gap-2 rounded-lg border px-3 py-2.5 text-[11px] ${
                      wizardStep === "done"
                        ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-300"
                        : "border-red-400/20 bg-red-400/10 text-red-300"
                    }`}>
                      {wizardStep === "done"
                        ? <Check className="w-4 h-4 shrink-0 mt-0.5" />
                        : <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />}
                      <span>
                        {wizardStep === "done"
                          ? `Node "${wizardName}" deployed successfully! It will appear as ONLINE within ~30 seconds once the agent connects.`
                          : wizardError || "Deployment failed."}
                      </span>
                    </div>
                    {wizardLog.length > 0 && (
                      <div className="rounded-lg bg-cs-surface border border-cs-border p-3 text-[10px] font-mono text-cs-dim space-y-0.5 max-h-48 overflow-y-auto">
                        {wizardLog.map((line, i) => (
                          <div key={i}>{line}</div>
                        ))}
                      </div>
                    )}
                    <div className="flex gap-2">
                      <button type="button" onClick={resetWizard} className="px-3 py-1.5 rounded-lg border border-cs-border text-[11px] font-medium text-cs-muted hover:text-cs-text transition-colors">
                        Close
                      </button>
                      {wizardStep === "error" && (
                        <button type="button" onClick={() => setWizardStep("probed")} className="px-3 py-1.5 rounded-lg border border-cs-border text-[11px] font-medium text-cs-muted hover:text-cs-text transition-colors">
                          Back to GPU Selection
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* GPU Safety Section */}
      <div>
        <h2 className="section-title flex items-center gap-2">
          <Shield className="w-3.5 h-3.5 text-orange-400" />
          GPU Safety Limits
        </h2>
        <div className="card p-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            <NumberInput
              label="GPU Memory Limit"
              value={config.safety.gpu_memory_limit}
              onChange={(v) => updateSafety("gpu_memory_limit", v)}
              min={0.5}
              max={0.99}
              step={0.01}
              hint="VRAM fraction — sustained above this triggers the guard pipeline"
            />
            <NumberInput
              label="Warning Threshold"
              value={config.safety.gpu_warn_threshold}
              onChange={(v) => updateSafety("gpu_warn_threshold", v)}
              min={0.3}
              max={0.99}
              step={0.01}
              hint="Dashboard warning level"
            />
            <NumberInput
              label="Danger Threshold"
              value={config.safety.gpu_danger_threshold}
              onChange={(v) => updateSafety("gpu_danger_threshold", v)}
              min={0.5}
              max={0.99}
              step={0.01}
              hint="Emergency GPU process kill level"
            />
            <NumberInput
              label="GPU util sustain threshold"
              value={config.safety.gpu_compute_sustain_threshold}
              onChange={(v) => updateSafety("gpu_compute_sustain_threshold", v)}
              min={0.5}
              max={0.99}
              step={0.01}
              hint="nvidia-smi GPU-Util % / 100 — must stay at or above this"
            />
            <NumberInput
              label="GPU util sustain duration (s)"
              value={config.safety.gpu_compute_sustain_duration_s}
              onChange={(v) => updateSafety("gpu_compute_sustain_duration_s", v)}
              min={10}
              max={600}
              step={1}
              hint="Wall time above threshold before guard treats it as a breach"
            />
            <NumberInput
              label="Mitigation Window (s)"
              value={config.safety.guard_mitigation_window_s}
              onChange={(v) => updateSafety("guard_mitigation_window_s", v)}
              min={60}
              max={3600}
              hint="Time to wait for self-healing before migrating"
            />
            <NumberInput
              label="Guard Check Interval (s)"
              value={config.safety.guard_check_interval_s}
              onChange={(v) => updateSafety("guard_check_interval_s", v)}
              min={5}
              max={120}
              hint="How often to sample VRAM and GPU utilization"
            />
          </div>
        </div>
      </div>

      {/* Per-Model Config */}
      <div>
        <h2 className="section-title flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-cs-accent2/70" />
          Model Configuration
        </h2>
        <div className="space-y-3">
          {Object.entries(config.models).map(([name, mcfg]) => {
            const expanded = expandedModels.has(name);
            const engineChanged = hasEngineChanges(name);
            return (
              <div key={name} className="card overflow-hidden">
                <button
                  onClick={() => toggleModel(name)}
                  className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-cs-hover transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {expanded ? (
                      <ChevronDown className="w-3.5 h-3.5 text-cs-dim" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 text-cs-dim" />
                    )}
                    <span className="text-[13px] font-semibold">{name}</span>
                    {engineChanged && (
                      <span className="badge bg-cs-warn/10 text-cs-warn border border-cs-warn/20">
                        Restart Required
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-cs-dim font-mono">
                      seqs={mcfg.engine.max_num_seqs} · vram=
                      {Math.round(mcfg.engine.gpu_memory_utilization * 100)}%
                    </span>
                    {engineChanged && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRestart(name);
                        }}
                        disabled={restartingModel === name}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-medium bg-cs-accent/10 text-cs-accent hover:bg-cs-accent/20 transition-colors border border-cs-accent/20"
                      >
                        {restartingModel === name ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <RefreshCw className="w-3 h-3" />
                        )}
                        Rolling Restart
                      </button>
                    )}
                  </div>
                </button>

                {expanded && (
                  <div className="px-5 pb-5 space-y-5 border-t border-cs-border/50 pt-4">
                    {/* Engine Settings */}
                    <div>
                      <h3 className="text-[10px] text-cs-dim font-semibold uppercase tracking-widest mb-3 flex items-center gap-2">
                        Engine Settings
                        <span className="text-cs-warn/60 normal-case tracking-normal font-normal">
                          (changes require restart)
                        </span>
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        <NumberInput
                          label="Max Num Seqs"
                          value={mcfg.engine.max_num_seqs}
                          onChange={(v) =>
                            updateEngine(name, "max_num_seqs", Math.round(v))
                          }
                          min={1}
                          max={256}
                          hint="Max concurrent sequences per replica"
                        />
                        <NumberInput
                          label="GPU Memory Utilization"
                          value={mcfg.engine.gpu_memory_utilization}
                          onChange={(v) =>
                            updateEngine(name, "gpu_memory_utilization", v)
                          }
                          min={0.3}
                          max={0.95}
                          step={0.05}
                          hint="Must stay below GPU safety limit"
                        />
                        <NumberInput
                          label="Max Model Length"
                          value={mcfg.engine.max_model_len}
                          onChange={(v) =>
                            updateEngine(name, "max_model_len", Math.round(v))
                          }
                          min={128}
                          max={131072}
                          step={1024}
                          hint="Context window size"
                        />
                      </div>
                      <div className="mt-3 space-y-1.5">
                        <ToggleInput
                          label="Chunked Prefill"
                          value={mcfg.engine.enable_chunked_prefill}
                          onChange={(v) =>
                            updateEngine(name, "enable_chunked_prefill", v)
                          }
                          hint="+8-10% throughput by interleaving prefill with decode"
                        />
                        <ToggleInput
                          label="Prefix Caching"
                          value={mcfg.engine.enable_prefix_caching}
                          onChange={(v) =>
                            updateEngine(name, "enable_prefix_caching", v)
                          }
                          hint="40-60% TTFT improvement for multi-turn conversations"
                        />
                      </div>
                    </div>

                    {/* Autoscaling Settings */}
                    <div>
                      <h3 className="text-[10px] text-cs-dim font-semibold uppercase tracking-widest mb-3 flex items-center gap-2">
                        Autoscaling Settings
                        <span className="text-cs-accent/40 normal-case tracking-normal font-normal">
                          (applied immediately)
                        </span>
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        <NumberInput
                          label="Min Replicas"
                          value={mcfg.autoscaling.min_replicas}
                          onChange={(v) =>
                            updateAutoscale(name, "min_replicas", Math.round(v))
                          }
                          min={0}
                          max={mcfg.autoscaling.max_replicas}
                        />
                        <NumberInput
                          label="Max Replicas"
                          value={mcfg.autoscaling.max_replicas}
                          onChange={(v) =>
                            updateAutoscale(name, "max_replicas", Math.round(v))
                          }
                          min={mcfg.autoscaling.min_replicas}
                          max={32}
                        />
                        <NumberInput
                          label="Target Inflight"
                          value={mcfg.autoscaling.target_inflight}
                          onChange={(v) =>
                            updateAutoscale(name, "target_inflight", v)
                          }
                          min={0.5}
                          max={100}
                          step={0.5}
                          hint="Avg inflight per replica trigger"
                        />
                        <NumberInput
                          label="Idle Timeout (s)"
                          value={mcfg.autoscaling.idle_timeout_s}
                          onChange={(v) =>
                            updateAutoscale(name, "idle_timeout_s", v)
                          }
                          min={10}
                          max={3600}
                          hint="Wait time before scale-down"
                        />
                        <NumberInput
                          label="Upscale Cooldown (s)"
                          value={mcfg.autoscaling.upscale_cooldown_s}
                          onChange={(v) =>
                            updateAutoscale(name, "upscale_cooldown_s", v)
                          }
                          min={5}
                          max={600}
                        />
                        <NumberInput
                          label="Downscale Cooldown (s)"
                          value={mcfg.autoscaling.downscale_cooldown_s}
                          onChange={(v) =>
                            updateAutoscale(name, "downscale_cooldown_s", v)
                          }
                          min={10}
                          max={600}
                        />
                        <NumberInput
                          label="Max Queue Depth"
                          value={mcfg.autoscaling.max_queue_depth}
                          onChange={(v) =>
                            updateAutoscale(
                              name,
                              "max_queue_depth",
                              Math.round(v),
                            )
                          }
                          min={0}
                          max={10000}
                          hint="0 = disabled (429 backpressure)"
                        />
                        <NumberInput
                          label="Replica Startup Timeout (s)"
                          value={mcfg.autoscaling.replica_startup_timeout_s}
                          onChange={(v) =>
                            updateAutoscale(name, "replica_startup_timeout_s", v)
                          }
                          min={60}
                          max={7200}
                          hint="vLLM /health wait during launch"
                        />
                        <div className="space-y-1">
                          <label className="text-[11px] text-cs-muted font-medium">
                            Allow Scale to Zero
                          </label>
                          <label className="flex items-center gap-2 text-[11px] text-cs-text cursor-pointer">
                            <input
                              type="checkbox"
                              checked={!!mcfg.autoscaling.allow_scale_to_zero}
                              onChange={(e) =>
                                updateAutoscale(name, "allow_scale_to_zero", e.target.checked)
                              }
                              className="rounded border-cs-border2"
                            />
                            Permit min replicas = 0 when idle
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Restart Warning Box */}
      <ControlPlaneLogsPanel />

      <div className="card p-5 border-cs-warn/20 bg-cs-warn/[0.02]">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-cs-warn mt-0.5 shrink-0" />
          <div className="space-y-2">
            <p className="text-[13px] font-semibold text-cs-warn">
              About Restarts
            </p>
            <p className="text-[11px] text-cs-muted leading-relaxed">
              <strong className="text-cs-text">Autoscaling changes</strong> take
              effect immediately — no restart needed.
            </p>
            <p className="text-[11px] text-cs-muted leading-relaxed">
              <strong className="text-cs-text">Engine changes</strong> (max_num_seqs,
              gpu_memory_utilization, chunked prefill, prefix caching) require a
              rolling restart. The system will:
            </p>
            <ol className="text-[11px] text-cs-muted list-decimal ml-4 space-y-1">
              <li>
                Set the replica to <span className="text-blue-400">DRAINING</span>{" "}
                — the gateway stops routing new requests to it
              </li>
              <li>
                Wait for all inflight requests to complete (
                <span className="text-cs-accent">zero request loss</span>)
              </li>
              <li>Stop the old vLLM process and launch a new one with updated settings</li>
              <li>
                Other replicas of the same model continue serving throughout
              </li>
            </ol>
            <p className="text-[11px] text-cs-muted leading-relaxed">
              <strong className="text-cs-text">GPU Safety Limit</strong> is an
              org-level hard cap (default 95%). Changing this affects all models.
              Setting{" "}
              <code className="text-cs-accent bg-cs-surface px-1 rounded">
                gpu_memory_utilization
              </code>{" "}
              above the safety limit will cause the GPU guard to kill replicas.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
