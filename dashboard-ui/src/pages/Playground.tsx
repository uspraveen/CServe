import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AudioLines,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  FlaskConical,
  ImagePlus,
  KeyRound,
  Loader2,
  MessageSquareText,
  Play,
  RefreshCw,
  Sparkles,
  Trash2,
  Upload,
  XCircle,
  Zap,
} from "lucide-react";
import type { ClusterSnapshot, ModelConfigInfo, ReplicaInfo } from "../hooks/useCluster";

type PlaygroundMode = "chat" | "embeddings" | "transcription";
type ChatRole = "user" | "assistant";

interface ChatTurn {
  id: string;
  role: ChatRole;
  content: string;
  imageDataUrl?: string;
  imageName?: string;
}

interface RunState {
  rendered: string;
  raw: string;
  error: string | null;
  statusCode: number | null;
  latencyMs: number | null;
  retryAfter: string | null;
  requestPath: string;
}

const STORAGE_KEY = "cserve.playground.apiKey";
const REMEMBER_KEY = "cserve.playground.remember";

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function defaultChatTurns(): ChatTurn[] {
  return [{ id: makeId(), role: "user", content: "Give me a concise summary of what this model is best suited for." }];
}

function modeOptions(config: ModelConfigInfo | undefined): PlaygroundMode[] {
  const caps = config?.capabilities || [];
  if (caps.includes("embeddings")) return ["embeddings"];
  if (caps.includes("transcription")) return ["transcription"];
  return ["chat"];
}

function prettyBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function extractChatText(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const choice = (payload as { choices?: Array<{ message?: { content?: unknown } }> }).choices?.[0];
  const content = choice?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object" && "text" in part) return String((part as { text?: unknown }).text ?? "");
      return "";
    }).join("");
  }
  return "";
}

function getModelReplicas(snapshot: ClusterSnapshot | null, modelName: string): ReplicaInfo[] {
  if (!snapshot) return [];
  return snapshot.replicas.filter((r) => r.model === modelName);
}

function getModelStatus(snapshot: ClusterSnapshot | null, modelName: string) {
  const replicas = getModelReplicas(snapshot, modelName);
  const ready   = replicas.filter((r) => r.status === "READY").length;
  const starting = replicas.filter((r) => r.status === "STARTING").length;
  const failed   = replicas.filter((r) => r.status === "FAILED").length;
  const total    = replicas.length;
  if (ready > 0 && failed === 0) return { label: "UP",           dot: "bg-emerald-400", tone: "text-emerald-400 border-emerald-400/20 bg-emerald-400/10", detail: `${ready} ready` };
  if (ready > 0)                 return { label: "DEGRADED",     dot: "bg-amber-400",   tone: "text-amber-400 border-amber-400/20 bg-amber-400/10",         detail: `${ready}/${total} ready` };
  if (starting > 0)              return { label: "STARTING",     dot: "bg-sky-400",     tone: "text-sky-400 border-sky-400/20 bg-sky-400/10",               detail: `${starting} launching` };
  if (total === 0)               return { label: "SCALED TO 0",  dot: "bg-cs-dim",      tone: "text-cs-dim border-cs-border2 bg-cs-border/60",              detail: "cold-start on demand" };
  return                                { label: "DOWN",          dot: "bg-red-400",     tone: "text-red-400 border-red-400/20 bg-red-400/10",               detail: `${failed || total} failed` };
}

async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

function CapabilityBadge({ label }: { label: string }) {
  const color = label === "vision"        ? "bg-sky-400/10 text-sky-300 border-sky-400/20"
              : label === "embeddings"    ? "bg-violet-400/10 text-violet-300 border-violet-400/20"
              : label === "transcription" ? "bg-amber-400/10 text-amber-300 border-amber-400/20"
              :                            "bg-cs-accent/10 text-cs-accent border-cs-accent/20";
  return <span className={`badge border ${color}`}>{label}</span>;
}

function ModeTab({ mode, active, onClick }: { mode: PlaygroundMode; active: boolean; onClick: () => void }) {
  const Icon = mode === "embeddings" ? BrainCircuit : mode === "transcription" ? AudioLines : MessageSquareText;
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-all duration-150 ${
        active ? "bg-cs-accent/10 text-cs-accent border border-cs-accent/20 shadow-glow-sm"
               : "text-cs-muted hover:text-cs-text border border-transparent"
      }`}
    >
      <Icon className="w-3.5 h-3.5" />
      <span className="capitalize">{mode}</span>
    </button>
  );
}

export default function Playground({ snapshot }: { snapshot: ClusterSnapshot | null }) {
  const [apiKey, setApiKey]                         = useState("");
  const [rememberKey, setRememberKey]               = useState(false);
  const [selectedModel, setSelectedModel]           = useState("");
  const [activeMode, setActiveMode]                 = useState<PlaygroundMode>("chat");
  const [systemPrompt, setSystemPrompt]             = useState("You are a precise, production-grade assistant.");
  const [chatTurns, setChatTurns]                   = useState<ChatTurn[]>(defaultChatTurns);
  const [temperature, setTemperature]               = useState(0.4);
  const [maxTokens, setMaxTokens]                   = useState(512);
  const [stream, setStream]                         = useState(true);
  const [embeddingInput, setEmbeddingInput]         = useState("FastAPI powers the gateway.\nRedis backs elastic queueing.\nvLLM serves the model replicas.");
  const [splitEmbeddingLines, setSplitEmbeddingLines] = useState(true);
  const [transcriptionFile, setTranscriptionFile]   = useState<File | null>(null);
  const [transcriptionPrompt, setTranscriptionPrompt] = useState("");
  const [transcriptionLanguage, setTranscriptionLanguage] = useState("");
  const [transcriptionFormat, setTranscriptionFormat] = useState("json");
  const [runState, setRunState]                     = useState<RunState>({ rendered: "", raw: "", error: null, statusCode: null, latencyMs: null, retryAfter: null, requestPath: "" });
  const [running, setRunning]                       = useState(false);
  const [rawTab, setRawTab]                         = useState<"rendered" | "raw">("rendered");
  const [copyState, setCopyState]                   = useState<"idle" | "request" | "response">("idle");
  const [previewOpen, setPreviewOpen]               = useState(false);

  const responseBodyRef = useRef<HTMLDivElement>(null);

  const modelConfigs = snapshot?.model_configs || {};
  const modelNames   = useMemo(() => Object.keys(modelConfigs).sort(), [modelConfigs]);

  useEffect(() => {
    const remember = window.localStorage.getItem(REMEMBER_KEY) === "1";
    setRememberKey(remember);
    if (remember) setApiKey(window.localStorage.getItem(STORAGE_KEY) || "");
  }, []);

  useEffect(() => {
    if (rememberKey) {
      window.localStorage.setItem(REMEMBER_KEY, "1");
      window.localStorage.setItem(STORAGE_KEY, apiKey);
    } else {
      window.localStorage.removeItem(REMEMBER_KEY);
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, [apiKey, rememberKey]);

  useEffect(() => {
    if (!selectedModel && modelNames.length > 0) setSelectedModel(modelNames[0]);
  }, [modelNames, selectedModel]);

  const selectedConfig   = selectedModel ? modelConfigs[selectedModel] : undefined;
  const supportedModes   = modeOptions(selectedConfig);
  const selectedStatus   = getModelStatus(snapshot, selectedModel);
  const modelQueue       = selectedModel && snapshot ? (snapshot.queue_depths[selectedModel] || 0) : 0;
  const modelReplicas    = selectedModel ? getModelReplicas(snapshot, selectedModel) : [];
  const readyReplicas    = modelReplicas.filter((r) => r.status === "READY").length;
  const isVisionModel    = Boolean(selectedConfig?.capabilities.includes("vision"));

  useEffect(() => {
    if (!supportedModes.includes(activeMode)) setActiveMode(supportedModes[0] || "chat");
  }, [activeMode, supportedModes]);

  // Auto-scroll response panel to bottom as tokens arrive
  useEffect(() => {
    if (running && responseBodyRef.current) {
      responseBodyRef.current.scrollTop = responseBodyRef.current.scrollHeight;
    }
  }, [runState.rendered, running]);

  const requestPreview = useMemo(() => {
    if (!selectedConfig) return "";
    if (activeMode === "embeddings") {
      const lines = embeddingInput.split("\n").map((l) => l.trim()).filter(Boolean);
      return JSON.stringify({ endpoint: "/v1/embeddings", headers: { Authorization: "Bearer csk_...", "Content-Type": "application/json" }, body: { model: selectedModel, input: splitEmbeddingLines ? (lines.length > 1 ? lines : lines[0] || "") : embeddingInput } }, null, 2);
    }
    if (activeMode === "transcription") {
      return JSON.stringify({ endpoint: "/v1/audio/transcriptions", headers: { Authorization: "Bearer csk_...", "Content-Type": "multipart/form-data" }, fields: { model: selectedModel, prompt: transcriptionPrompt || undefined, language: transcriptionLanguage || undefined, response_format: transcriptionFormat }, file: transcriptionFile ? { name: transcriptionFile.name, size: prettyBytes(transcriptionFile.size), type: transcriptionFile.type || "application/octet-stream" } : null }, null, 2);
    }
    const messages = [
      ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
      ...chatTurns.filter((t) => t.content.trim() || t.imageDataUrl).map((t) => ({
        role: t.role,
        content: t.imageDataUrl
          ? [...(t.content.trim() ? [{ type: "text", text: t.content.trim() }] : []), { type: "image_url", image_url: { url: `<data-url ${Math.round(t.imageDataUrl.length / 1024)} KiB>` } }]
          : t.content.trim(),
      })),
    ];
    return JSON.stringify({ endpoint: "/v1/chat/completions", headers: { Authorization: "Bearer csk_...", "Content-Type": "application/json" }, body: { model: selectedModel, messages, temperature, max_tokens: maxTokens, stream } }, null, 2);
  }, [activeMode, chatTurns, embeddingInput, maxTokens, selectedConfig, selectedModel, splitEmbeddingLines, stream, systemPrompt, temperature, transcriptionFile, transcriptionFormat, transcriptionLanguage, transcriptionPrompt]);

  const resetRunState = () => {
    setRunState({ rendered: "", raw: "", error: null, statusCode: null, latencyMs: null, retryAfter: null, requestPath: "" });
    setRawTab("rendered");
  };

  const handleCopy = async (value: string, kind: "request" | "response") => {
    try { await navigator.clipboard.writeText(value); setCopyState(kind); setTimeout(() => setCopyState("idle"), 1200); }
    catch { setCopyState("idle"); }
  };

  const addTurn    = (role: ChatRole) => setChatTurns((p) => [...p, { id: makeId(), role, content: "" }]);
  const updateTurn = (id: string, patch: Partial<ChatTurn>) => setChatTurns((p) => p.map((t) => t.id === id ? { ...t, ...patch } : t));
  const removeTurn = (id: string) => setChatTurns((p) => p.length === 1 ? p : p.filter((t) => t.id !== id));
  const attachImage = async (turnId: string, file: File | null) => {
    if (!file) return;
    const dataUrl = await fileToDataUrl(file);
    updateTurn(turnId, { imageDataUrl: dataUrl, imageName: file.name });
  };

  const execute = async () => {
    if (!selectedConfig || !selectedModel) { setRunState((p) => ({ ...p, error: "No model selected." })); return; }
    if (!apiKey.trim()) { setRunState((p) => ({ ...p, error: "Enter an API key to use the playground." })); return; }

    resetRunState();
    setRunning(true);
    setRawTab("rendered");
    const startedAt = performance.now();

    try {
      // ── Embeddings ─────────────────────────────────────────────────────────
      if (activeMode === "embeddings") {
        const lines = embeddingInput.split("\n").map((l) => l.trim()).filter(Boolean);
        const input = splitEmbeddingLines ? (lines.length > 1 ? lines : lines[0] || "") : embeddingInput;
        if (!input || (Array.isArray(input) && input.length === 0)) throw new Error("Add one or more texts to embed.");

        const resp = await fetch("/v1/embeddings", { method: "POST", headers: { Authorization: `Bearer ${apiKey.trim()}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: selectedModel, input }) });
        const latencyMs = performance.now() - startedAt;
        const rawText = await resp.text();
        let parsed: unknown = rawText;
        try { parsed = JSON.parse(rawText); } catch { /* keep string */ }

        if (!resp.ok) {
          const msg = typeof parsed === "object" && parsed && "error" in parsed ? String(((parsed as { error?: { message?: string } }).error?.message) || "Request failed") : `Embedding request failed (${resp.status})`;
          setRunState({ rendered: "", raw: typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2), error: msg, statusCode: resp.status, latencyMs, retryAfter: resp.headers.get("Retry-After"), requestPath: "/v1/embeddings" });
          return;
        }
        const vectorCount = Array.isArray((parsed as { data?: unknown[] })?.data) ? (parsed as { data: unknown[] }).data.length : 0;
        const firstVector = (parsed as { data?: Array<{ embedding?: number[] }> })?.data?.[0]?.embedding || [];
        const rendered = vectorCount > 0 ? `${vectorCount} embedding${vectorCount === 1 ? "" : "s"} generated\n\nDimensions: ${firstVector.length || 0}\nPreview: ${firstVector.slice(0, 12).map((n) => n.toFixed(4)).join(", ")}` : "Embedding response received.";
        setRunState({ rendered, raw: typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2), error: null, statusCode: resp.status, latencyMs, retryAfter: resp.headers.get("Retry-After"), requestPath: "/v1/embeddings" });
        return;
      }

      // ── Transcription ───────────────────────────────────────────────────────
      if (activeMode === "transcription") {
        if (!transcriptionFile) throw new Error("Attach an audio file to transcribe.");
        const formData = new FormData();
        formData.set("model", selectedModel);
        formData.set("file", transcriptionFile);
        if (transcriptionPrompt.trim())   formData.set("prompt",   transcriptionPrompt.trim());
        if (transcriptionLanguage.trim()) formData.set("language", transcriptionLanguage.trim());
        formData.set("response_format", transcriptionFormat);

        const resp = await fetch("/v1/audio/transcriptions", { method: "POST", headers: { Authorization: `Bearer ${apiKey.trim()}` }, body: formData });
        const latencyMs = performance.now() - startedAt;
        const ct = resp.headers.get("content-type") || "";
        const rawText = await resp.text();
        let parsed: unknown = rawText;
        if (ct.includes("application/json")) { try { parsed = JSON.parse(rawText); } catch { /* keep string */ } }

        if (!resp.ok) {
          const msg = typeof parsed === "object" && parsed && "error" in parsed ? String(((parsed as { error?: { message?: string } }).error?.message) || "Request failed") : `Transcription request failed (${resp.status})`;
          setRunState({ rendered: "", raw: typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2), error: msg, statusCode: resp.status, latencyMs, retryAfter: resp.headers.get("Retry-After"), requestPath: "/v1/audio/transcriptions" });
          return;
        }
        const rendered = typeof parsed === "string" ? parsed : String((parsed as { text?: string }).text || "Transcription completed.");
        setRunState({ rendered, raw: typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2), error: null, statusCode: resp.status, latencyMs, retryAfter: resp.headers.get("Retry-After"), requestPath: "/v1/audio/transcriptions" });
        return;
      }

      // ── Chat / Vision ───────────────────────────────────────────────────────
      const messages = [
        ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
        ...chatTurns.filter((t) => t.content.trim() || t.imageDataUrl).map((t) => ({
          role: t.role,
          content: t.imageDataUrl
            ? [...(t.content.trim() ? [{ type: "text", text: t.content.trim() }] : []), { type: "image_url", image_url: { url: t.imageDataUrl } }]
            : t.content.trim(),
        })),
      ];
      if (messages.length === 0) throw new Error("Add at least one chat message.");

      const resp = await fetch("/v1/chat/completions", { method: "POST", headers: { Authorization: `Bearer ${apiKey.trim()}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: selectedModel, messages, temperature, max_tokens: maxTokens, stream }) });

      if (!resp.ok) {
        const latencyMs = performance.now() - startedAt;
        const rawText = await resp.text();
        let parsed: unknown = rawText;
        try { parsed = JSON.parse(rawText); } catch { /* keep string */ }
        const msg = typeof parsed === "object" && parsed && "error" in parsed ? String(((parsed as { error?: { message?: string } }).error?.message) || "Request failed") : `Chat request failed (${resp.status})`;
        setRunState({ rendered: "", raw: typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2), error: msg, statusCode: resp.status, latencyMs, retryAfter: resp.headers.get("Retry-After"), requestPath: "/v1/chat/completions" });
        return;
      }

      if (!stream) {
        const latencyMs = performance.now() - startedAt;
        const parsed = await resp.json();
        setRunState({ rendered: extractChatText(parsed), raw: JSON.stringify(parsed, null, 2), error: null, statusCode: resp.status, latencyMs, retryAfter: resp.headers.get("Retry-After"), requestPath: "/v1/chat/completions" });
        return;
      }

      if (!resp.body) throw new Error("Streaming response body is unavailable.");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "", assembled = "";
      const chunks: unknown[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const event = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          for (const line of event.split("\n").map((l) => l.trim()).filter(Boolean)) {
            if (!line.startsWith("data:")) continue;
            const raw = line.slice(5).trim();
            if (!raw || raw === "[DONE]") continue;
            try {
              const parsed = JSON.parse(raw);
              chunks.push(parsed);
              const delta = parsed.choices?.[0]?.delta?.content;
              if (typeof delta === "string") assembled += delta;
              else if (Array.isArray(delta)) assembled += delta.map((p: { text?: string }) => p?.text || "").join("");
              setRunState({ rendered: assembled, raw: JSON.stringify(chunks, null, 2), error: null, statusCode: resp.status, latencyMs: performance.now() - startedAt, retryAfter: resp.headers.get("Retry-After"), requestPath: "/v1/chat/completions" });
            } catch { /* ignore malformed SSE */ }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
      setRunState((p) => ({ ...p, rendered: p.rendered || assembled || "Stream completed.", raw: p.raw || JSON.stringify(chunks, null, 2), latencyMs: performance.now() - startedAt }));
    } catch (error) {
      setRunState({ rendered: "", raw: "", error: error instanceof Error ? error.message : String(error), statusCode: null, latencyMs: performance.now() - startedAt, retryAfter: null, requestPath: activeMode === "embeddings" ? "/v1/embeddings" : activeMode === "transcription" ? "/v1/audio/transcriptions" : "/v1/chat/completions" });
    } finally {
      setRunning(false);
    }
  };

  const endpointLabel = activeMode === "embeddings" ? "/v1/embeddings" : activeMode === "transcription" ? "/v1/audio/transcriptions" : "/v1/chat/completions";

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-5 animate-fade-in">

      {/* ── Page header ── */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <FlaskConical className="w-4 h-4 text-cs-accent" />
            <h1 className="text-[15px] font-semibold tracking-tight">Playground</h1>
          </div>
          <p className="text-[11px] text-cs-dim">
            Live requests through the CServe gateway — chat, vision, embeddings, transcription.
          </p>
        </div>
        <div className="flex items-center gap-4 text-[11px] font-mono text-cs-dim">
          <span className="text-cs-muted">{modelNames.length} models</span>
          <span><span className="text-cs-accent font-semibold">{snapshot?.stats.ready_replicas ?? 0}</span> ready</span>
          <span className="rounded-lg border border-cs-border bg-cs-card px-2.5 py-1 text-cs-dim">{endpointLabel}</span>
        </div>
      </div>

      {/* ── Session bar — API key + model + mode ── */}
      <div className="card p-4">
        <div className="grid gap-4 md:grid-cols-[1fr_1fr_auto]">
          {/* API key */}
          <div className="space-y-1.5">
            <div className="section-title mb-0 flex items-center gap-1.5">
              <KeyRound className="w-3 h-3" /> API Key
            </div>
            <div className="space-y-1">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="csk_..."
                className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2 text-sm font-mono text-cs-text outline-none transition-all focus:border-cs-accent/40 focus:shadow-glow-sm"
              />
              <label className="flex items-center gap-1.5 text-[10px] text-cs-dim cursor-pointer select-none">
                <input type="checkbox" checked={rememberKey} onChange={(e) => setRememberKey(e.target.checked)} className="rounded border-cs-border bg-cs-surface text-cs-accent focus:ring-cs-accent/30" />
                Remember in browser
              </label>
            </div>
          </div>

          {/* Model selector */}
          <div className="space-y-1.5">
            <div className="section-title mb-0">Model</div>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2 text-sm text-cs-text outline-none transition-all focus:border-cs-accent/40"
            >
              {modelNames.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            {selectedConfig && (
              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${selectedStatus.tone}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${selectedStatus.dot}`} />
                  {selectedStatus.label}
                </span>
                <span className="text-[10px] text-cs-dim font-mono truncate max-w-[160px]">{selectedConfig.hf_model}</span>
                {selectedConfig.capabilities.map((c) => <CapabilityBadge key={c} label={c} />)}
              </div>
            )}
          </div>

          {/* Mode tabs + model stats */}
          <div className="space-y-1.5">
            <div className="section-title mb-0">Mode</div>
            <div className="flex gap-1">
              {supportedModes.map((mode) => (
                <ModeTab key={mode} mode={mode} active={activeMode === mode} onClick={() => setActiveMode(mode)} />
              ))}
            </div>
            {selectedConfig && (
              <div className="flex items-center gap-3 text-[10px] font-mono text-cs-dim">
                <span><span className="text-cs-text">{readyReplicas}</span> ready</span>
                <span><span className="text-cs-text">{modelQueue}</span> queued</span>
                <span className="text-cs-dim">{selectedConfig.autoscaling.min_replicas}–{selectedConfig.autoscaling.max_replicas} replicas</span>
              </div>
            )}
          </div>
        </div>

        {/* Scale-to-zero notice */}
        {selectedConfig?.autoscaling.min_replicas === 0 && (
          <div className="mt-3 rounded-lg border border-amber-400/20 bg-amber-400/8 px-3 py-2 text-[11px] text-amber-200/80">
            <Zap className="inline w-3.5 h-3.5 mr-1" />
            This model can scale to zero — the first request will trigger a cold start and return 503 briefly while a replica launches.
          </div>
        )}
      </div>

      {/* ── Main split: Composer | Response — both start at the same level ── */}
      <div className="grid gap-5 lg:grid-cols-2">

        {/* LEFT: Composer */}
        <div className="card flex flex-col">
          <div className="flex items-center justify-between border-b border-cs-border px-5 py-3.5">
            <div className="flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-cs-accent2/70" />
              <span className="text-[13px] font-semibold">Composer</span>
            </div>
            <div className="flex items-center gap-1">
              {supportedModes.map((mode) => (
                <ModeTab key={mode} mode={mode} active={activeMode === mode} onClick={() => setActiveMode(mode)} />
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-4">

            {/* ── Chat mode ── */}
            {activeMode === "chat" && (
              <>
                <div className="space-y-1.5">
                  <div className="section-title mb-0">System Prompt</div>
                  <textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    rows={2}
                    className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2.5 text-sm text-cs-text outline-none transition-all focus:border-cs-accent/40 resize-none"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <div className="section-title mb-0">Temperature</div>
                    <input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value) || 0)}
                      className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2 text-sm font-mono text-cs-text outline-none focus:border-cs-accent/40" />
                  </div>
                  <div className="space-y-1.5">
                    <div className="section-title mb-0">Max Tokens</div>
                    <input type="number" min={1} step={1} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value) || 1)}
                      className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2 text-sm font-mono text-cs-text outline-none focus:border-cs-accent/40" />
                  </div>
                </div>

                <label className="flex items-center justify-between rounded-lg border border-cs-border bg-cs-surface/50 px-3 py-2.5 cursor-pointer">
                  <div>
                    <span className="text-sm font-medium text-cs-text">Stream tokens live</span>
                    <span className="ml-2 text-[10px] text-cs-dim">SSE</span>
                  </div>
                  <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} className="rounded border-cs-border bg-cs-surface text-cs-accent focus:ring-cs-accent/30" />
                </label>

                <div className="space-y-2">
                  <div className="section-title mb-0">Messages</div>
                  {chatTurns.map((turn, index) => (
                    <div key={turn.id} className="rounded-lg border border-cs-border bg-cs-surface/60 p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <select value={turn.role} onChange={(e) => updateTurn(turn.id, { role: e.target.value as ChatRole })}
                            className="rounded border border-cs-border2 bg-cs-card px-2 py-1 text-[10px] uppercase tracking-[0.16em] text-cs-text outline-none">
                            <option value="user">user</option>
                            <option value="assistant">assistant</option>
                          </select>
                          <span className="text-[10px] text-cs-dim">Turn {index + 1}</span>
                        </div>
                        <button type="button" onClick={() => removeTurn(turn.id)} className="rounded p-1 text-cs-dim transition-colors hover:text-red-300">
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <textarea value={turn.content} onChange={(e) => updateTurn(turn.id, { content: e.target.value })} rows={3}
                        placeholder={turn.role === "user" ? "Ask a question or describe the image." : "Optional prior assistant message."}
                        className="w-full rounded-lg border border-cs-border2 bg-cs-card px-3 py-2 text-sm text-cs-text outline-none transition-all focus:border-cs-accent/40 resize-none" />
                      {isVisionModel && turn.role === "user" && (
                        <div className="rounded-lg border border-dashed border-cs-border2 bg-cs-card/70 px-3 py-2.5">
                          <div className="flex items-center justify-between gap-3">
                            <span className="text-[11px] text-cs-dim">Image attachment</span>
                            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-cs-border2 bg-cs-surface px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cs-text transition-colors hover:border-cs-accent/30">
                              <ImagePlus className="w-3.5 h-3.5" /> Attach
                              <input type="file" accept="image/*" className="hidden" onChange={async (e) => { const f = e.target.files?.[0] || null; if (f) await attachImage(turn.id, f); }} />
                            </label>
                          </div>
                          {turn.imageName && (
                            <div className="mt-2 flex items-center justify-between rounded border border-sky-400/20 bg-sky-400/10 px-2.5 py-1.5 text-[10px] text-sky-200">
                              <span className="truncate">{turn.imageName}</span>
                              <button type="button" onClick={() => updateTurn(turn.id, { imageDataUrl: undefined, imageName: undefined })} className="ml-2 hover:text-white">remove</button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                  <div className="flex gap-2">
                    <button type="button" onClick={() => addTurn("user")} className="rounded-lg border border-cs-border bg-cs-surface px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cs-text transition-colors hover:border-cs-accent/30">+ user</button>
                    <button type="button" onClick={() => addTurn("assistant")} className="rounded-lg border border-cs-border bg-cs-surface px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cs-text transition-colors hover:border-cs-accent/30">+ assistant</button>
                    <button type="button" onClick={() => setChatTurns(defaultChatTurns())} className="rounded-lg border border-cs-border bg-cs-surface px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cs-dim transition-colors hover:text-cs-text">reset</button>
                  </div>
                </div>
              </>
            )}

            {/* ── Embeddings mode ── */}
            {activeMode === "embeddings" && (
              <>
                <div className="space-y-1.5">
                  <div className="section-title mb-0">Input Text</div>
                  <p className="text-[10px] text-cs-dim">One text per line for batch embedding.</p>
                  <textarea value={embeddingInput} onChange={(e) => setEmbeddingInput(e.target.value)} rows={8}
                    className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2.5 text-sm text-cs-text outline-none transition-all focus:border-violet-400/40 resize-none" />
                </div>
                <label className="flex items-center justify-between rounded-lg border border-cs-border bg-cs-surface/50 px-3 py-2.5 cursor-pointer">
                  <div>
                    <span className="text-sm font-medium text-cs-text">Split lines into multiple inputs</span>
                    <div className="text-[10px] text-cs-dim">Each non-empty line becomes its own item.</div>
                  </div>
                  <input type="checkbox" checked={splitEmbeddingLines} onChange={(e) => setSplitEmbeddingLines(e.target.checked)} className="rounded border-cs-border bg-cs-surface text-violet-400 focus:ring-violet-400/30" />
                </label>
              </>
            )}

            {/* ── Transcription mode ── */}
            {activeMode === "transcription" && (
              <>
                <div className="rounded-lg border border-dashed border-cs-border2 bg-cs-surface/60 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-sm font-medium text-cs-text">Audio or video file</div>
                      <div className="mt-1 text-[10px] text-cs-dim">Multipart form upload via CServe gateway.</div>
                    </div>
                    <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-cs-border2 bg-cs-card px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-cs-text transition-colors hover:border-amber-400/30">
                      <Upload className="w-3.5 h-3.5" /> Upload
                      <input type="file" accept="audio/*,video/*" className="hidden" onChange={(e) => setTranscriptionFile(e.target.files?.[0] || null)} />
                    </label>
                  </div>
                  {transcriptionFile && (
                    <div className="mt-3 rounded-lg border border-amber-400/20 bg-amber-400/10 px-3 py-2 text-[10px] text-amber-200">
                      <div className="font-medium">{transcriptionFile.name}</div>
                      <div className="mt-0.5">{prettyBytes(transcriptionFile.size)} · {transcriptionFile.type || "unknown"}</div>
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <div className="section-title mb-0">Language</div>
                    <input value={transcriptionLanguage} onChange={(e) => setTranscriptionLanguage(e.target.value)} placeholder="en"
                      className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2 text-sm text-cs-text outline-none focus:border-amber-400/40" />
                  </div>
                  <div className="space-y-1.5">
                    <div className="section-title mb-0">Format</div>
                    <select value={transcriptionFormat} onChange={(e) => setTranscriptionFormat(e.target.value)}
                      className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2 text-sm text-cs-text outline-none focus:border-amber-400/40">
                      <option value="json">json</option>
                      <option value="text">text</option>
                      <option value="verbose_json">verbose_json</option>
                    </select>
                  </div>
                </div>
                <div className="space-y-1.5">
                  <div className="section-title mb-0">Prompt Hint</div>
                  <textarea value={transcriptionPrompt} onChange={(e) => setTranscriptionPrompt(e.target.value)} rows={3}
                    placeholder="Domain-specific terms, names, or vocabulary..."
                    className="w-full rounded-lg border border-cs-border bg-cs-surface px-3 py-2 text-sm text-cs-text outline-none focus:border-amber-400/40 resize-none" />
                </div>
              </>
            )}
          </div>

          {/* Run bar — always visible at the bottom of the composer */}
          <div className="border-t border-cs-border bg-cs-surface/30 px-5 py-3.5 flex items-center gap-3">
            <button
              type="button"
              onClick={execute}
              disabled={running || !selectedModel}
              className="inline-flex items-center gap-2 rounded-lg bg-cs-accent px-4 py-2.5 text-sm font-semibold text-black transition-all hover:bg-cs-accent/90 disabled:cursor-not-allowed disabled:opacity-60 shadow-glow-sm"
            >
              {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {running ? "Running…" : "Run"}
            </button>
            <button
              type="button"
              onClick={resetRunState}
              className="inline-flex items-center gap-2 rounded-lg border border-cs-border bg-cs-surface px-3 py-2.5 text-sm font-medium text-cs-muted transition-colors hover:text-cs-text"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Clear
            </button>
            {runState.latencyMs !== null && (
              <span className="ml-auto text-[11px] font-mono text-cs-dim">
                {runState.latencyMs.toFixed(0)} ms
              </span>
            )}
          </div>
        </div>

        {/* RIGHT: Response — starts at the same vertical level as the Composer */}
        <div className="card flex flex-col">
          <div className="flex items-center justify-between border-b border-cs-border px-5 py-3.5">
            <div className="flex items-center gap-3">
              <span className="text-[13px] font-semibold">Response</span>
              {runState.statusCode !== null && (
                <span className={`badge border ${runState.error ? "border-red-400/20 bg-red-400/10 text-red-300" : "border-emerald-400/20 bg-emerald-400/10 text-emerald-300"}`}>
                  HTTP {runState.statusCode}
                </span>
              )}
              {running && (
                <span className="badge border border-cs-accent/20 bg-cs-accent/10 text-cs-accent flex items-center gap-1">
                  <Loader2 className="w-3 h-3 animate-spin" /> streaming
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {/* Rendered / Raw tabs */}
              <div className="flex gap-1">
                {(["rendered", "raw"] as const).map((tab) => (
                  <button key={tab} type="button" onClick={() => setRawTab(tab)}
                    className={`rounded px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] transition-all ${
                      rawTab === tab
                        ? tab === "rendered" ? "bg-cs-accent/10 text-cs-accent" : "bg-cs-accent2/10 text-cs-accent2"
                        : "text-cs-dim hover:text-cs-text"
                    }`}>
                    {tab}
                  </button>
                ))}
              </div>
              <button type="button" onClick={() => handleCopy(runState.raw || runState.rendered, "response")}
                disabled={!runState.raw && !runState.rendered}
                className="inline-flex items-center gap-1.5 rounded-lg border border-cs-border bg-cs-surface px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cs-text transition-colors hover:border-cs-accent/30 disabled:opacity-40">
                <Copy className="w-3.5 h-3.5" />
                {copyState === "response" ? "copied" : "copy"}
              </button>
            </div>
          </div>

          {/* Response body — scrollable, tokens appear here */}
          <div ref={responseBodyRef} className="flex-1 overflow-y-auto p-5 min-h-[420px]">

            {runState.error && (
              <div className="mb-4 rounded-lg border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-200">
                <div className="flex items-center gap-2 font-semibold mb-1"><XCircle className="w-4 h-4" /> Request failed</div>
                <div className="leading-6">{runState.error}</div>
                {runState.retryAfter && <div className="mt-1.5 text-[10px] uppercase tracking-[0.16em] text-red-200/70">Retry-After: {runState.retryAfter}s</div>}
              </div>
            )}

            {!runState.error && runState.statusCode !== null && runState.statusCode < 400 && (
              <div className="mb-4 rounded-lg border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-[10px] text-emerald-200 flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span className="font-semibold uppercase tracking-[0.16em]">Completed</span>
                {runState.requestPath && <span className="font-mono text-emerald-200/60">{runState.requestPath}</span>}
              </div>
            )}

            {rawTab === "rendered" ? (
              runState.rendered ? (
                <pre className="whitespace-pre-wrap break-words rounded-lg border border-cs-border bg-cs-surface/70 p-4 text-sm leading-7 text-cs-text">
                  {runState.rendered}
                </pre>
              ) : (
                <div className="flex min-h-[300px] flex-col items-center justify-center rounded-lg border border-dashed border-cs-border2 bg-cs-surface/30 px-6 text-center">
                  <Bot className="mb-3 h-8 w-8 text-cs-dim" />
                  <div className="text-sm font-medium text-cs-text">No response yet</div>
                  <div className="mt-1.5 max-w-xs text-[11px] leading-6 text-cs-dim">
                    Compose a request and press <strong className="text-cs-text">Run</strong>. Streaming tokens appear here live.
                  </div>
                </div>
              )
            ) : (
              runState.raw ? (
                <pre className="overflow-x-auto rounded-lg border border-cs-border bg-cs-surface/70 p-4 text-[12px] leading-6 text-cs-text min-h-[300px]">
                  {runState.raw}
                </pre>
              ) : (
                <div className="flex min-h-[300px] flex-col items-center justify-center rounded-lg border border-dashed border-cs-border2 bg-cs-surface/30 px-6 text-center">
                  <Activity className="mb-3 h-8 w-8 text-cs-dim" />
                  <div className="text-sm font-medium text-cs-text">Raw payload</div>
                  <div className="mt-1.5 max-w-xs text-[11px] leading-6 text-cs-dim">The exact OpenAI-compatible wire format will appear here after a request.</div>
                </div>
              )
            )}
          </div>
        </div>
      </div>

      {/* ── Request Preview (collapsible) ── */}
      <div className="card overflow-hidden">
        <button
          type="button"
          onClick={() => setPreviewOpen((p) => !p)}
          className="w-full flex items-center justify-between px-5 py-3.5 text-left hover:bg-cs-hover transition-colors"
        >
          <div className="flex items-center gap-2">
            {previewOpen ? <ChevronDown className="w-3.5 h-3.5 text-cs-dim" /> : <ChevronRight className="w-3.5 h-3.5 text-cs-dim" />}
            <span className="text-[13px] font-semibold">Request Preview</span>
            <span className="text-[10px] text-cs-dim">client-facing payload sketch</span>
          </div>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); handleCopy(requestPreview, "request"); }}
            className="inline-flex items-center gap-1.5 rounded-lg border border-cs-border bg-cs-surface px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cs-text transition-colors hover:border-cs-accent/30"
            disabled={!requestPreview}
          >
            <Copy className="w-3.5 h-3.5" />
            {copyState === "request" ? "copied" : "copy"}
          </button>
        </button>
        {previewOpen && (
          <div className="border-t border-cs-border p-5">
            <pre className="overflow-x-auto rounded-lg border border-cs-border bg-cs-surface/70 p-4 text-[12px] leading-6 text-cs-text">
              {requestPreview || "// select a model to start"}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
