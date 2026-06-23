import { useCallback, useEffect, useMemo, useState } from "react";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";
import type { ClusterSnapshot, AutoscaleEvent } from "../hooks/useCluster";
import TimeRangeBar from "../components/TimeRangeBar";
import {
  type TimeRangePreset,
  defaultCustomInputs,
  localInputToUnix,
  presetWindowS,
} from "../lib/timeRange";

function ActionBadge({ action }: { action: string }) {
  if (action === "SCALE_UP")
    return (
      <span className="badge bg-cs-accent/10 text-cs-accent border border-cs-accent/20 flex items-center gap-0.5 w-fit">
        <ArrowUpRight className="w-3 h-3" />
        UP
      </span>
    );
  if (action === "SCALE_DOWN")
    return (
      <span className="badge bg-cs-warn/10 text-cs-warn border border-cs-warn/20 flex items-center gap-0.5 w-fit">
        <ArrowDownRight className="w-3 h-3" />
        DOWN
      </span>
    );
  if (action === "SCALE_TO_ZERO")
    return (
      <span className="badge bg-cs-danger/10 text-cs-danger border border-cs-danger/20 flex items-center gap-0.5 w-fit">
        <Minus className="w-3 h-3" />
        ZERO
      </span>
    );
  return null;
}

function formatEventTime(ts: number, preset: TimeRangePreset): string {
  const d = new Date(ts * 1000);
  if (preset !== "24h") {
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  return d.toLocaleTimeString();
}

function ModelScaleCard({
  model,
  replicas,
  events,
  hasLaunchFailures,
  preset,
}: {
  model: string;
  replicas: number;
  events: AutoscaleEvent[];
  hasLaunchFailures: boolean;
  preset: TimeRangePreset;
}) {
  const recent = events[0]?.action;
  const scalingUp = replicas === 0 && recent === "SCALE_UP";
  const failedRetrying = scalingUp && hasLaunchFailures;
  const borderCls =
    recent === "SCALE_UP"
      ? "border-cs-accent/20"
      : recent === "SCALE_DOWN"
        ? "border-cs-warn/20"
        : "border-cs-border";

  return (
    <div className={`card-hover p-5 space-y-4 ${borderCls}`}>
      <div className="flex items-center justify-between">
        <span className="text-[14px] font-semibold">{model}</span>
        <div className="flex items-baseline gap-1.5">
          {failedRetrying && (
            <span className="text-[10px] text-cs-danger font-medium mr-1.5 animate-pulse">
              FAILED – retrying
            </span>
          )}
          {scalingUp && !failedRetrying && (
            <span className="text-[10px] text-cs-accent font-medium mr-1.5 animate-pulse">
              Starting…
            </span>
          )}
          <span className="text-3xl font-bold font-mono text-cs-accent">
            {replicas}
          </span>
          <span className="text-[10px] text-cs-dim">replicas</span>
        </div>
      </div>

      {events.length > 0 && (
        <div className="space-y-1.5">
          {events.slice(0, 5).map((e, i) => (
            <div
              key={`${e.timestamp}-${i}`}
              className="flex items-center gap-2.5 text-xs"
            >
              <span className="text-cs-dim font-mono text-[10px] w-28 shrink-0">
                {formatEventTime(e.timestamp, preset)}
              </span>
              <ActionBadge action={e.action} />
              <span className="text-cs-muted font-mono">
                {e.from_replicas} → {e.to_replicas}
              </span>
              <span className="text-cs-dim truncate text-[10px]">
                {e.reasons.join(", ")}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface CustomEpoch {
  since: number;
  until: number;
}

export default function Autoscaling({
  snapshot,
}: {
  snapshot: ClusterSnapshot | null;
}) {
  const [preset, setPreset] = useState<TimeRangePreset>("24h");
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const [customEpoch, setCustomEpoch] = useState<CustomEpoch | null>(null);
  const [customError, setCustomError] = useState<string | null>(null);
  const [scaleEvents, setScaleEvents] = useState<AutoscaleEvent[]>([]);
  const [scaleLoading, setScaleLoading] = useState(true);

  const applyCustomRange = useCallback((startStr: string, endStr: string) => {
    const since = localInputToUnix(startStr);
    const until = localInputToUnix(endStr);
    if (!Number.isFinite(since) || !Number.isFinite(until)) {
      setCustomError("Invalid date/time");
      return;
    }
    if (until <= since) {
      setCustomError("End must be after start");
      return;
    }
    setCustomError(null);
    setCustomEpoch({ since, until });
  }, []);

  const handlePreset = useCallback(
    (p: TimeRangePreset) => {
      if (p === "custom") {
        const { start, end } = defaultCustomInputs();
        setCustomStart(start);
        setCustomEnd(end);
        setPreset("custom");
        applyCustomRange(start, end);
      } else {
        setPreset(p);
        setCustomEpoch(null);
        setCustomError(null);
      }
    },
    [applyCustomRange],
  );

  const fetchScaleEvents = useCallback(async () => {
    if (preset === "custom" && !customEpoch) {
      setScaleEvents([]);
      setScaleLoading(false);
      return;
    }
    let url = "/dashboard/api/events/autoscale?limit=2000&";
    if (preset === "custom" && customEpoch) {
      url += `since_ts=${customEpoch.since}&until_ts=${customEpoch.until}`;
    } else if (preset !== "custom") {
      url += `window_s=${presetWindowS(preset)}`;
    }
    setScaleLoading(true);
    try {
      const r = await fetch(url);
      const data = (await r.json()) as AutoscaleEvent[];
      setScaleEvents(Array.isArray(data) ? data : []);
    } catch {
      setScaleEvents([]);
    } finally {
      setScaleLoading(false);
    }
  }, [preset, customEpoch]);

  useEffect(() => {
    void fetchScaleEvents();
  }, [fetchScaleEvents]);

  useEffect(() => {
    const id = setInterval(() => void fetchScaleEvents(), 15000);
    return () => clearInterval(id);
  }, [fetchScaleEvents]);

  const eventsByModel = useMemo(() => {
    const map: Record<string, AutoscaleEvent[]> = {};
    const sorted = [...scaleEvents].sort((a, b) => b.timestamp - a.timestamp);
    for (const e of sorted) {
      (map[e.model] ??= []).push(e);
    }
    return map;
  }, [scaleEvents]);

  const modelReplicas = useMemo(() => {
    const map: Record<string, number> = {};
    if (!snapshot) return map;
    for (const r of snapshot.replicas) {
      map[r.model] = (map[r.model] || 0) + 1;
    }
    return map;
  }, [snapshot?.replicas]);

  const models = useMemo(() => {
    if (!snapshot) return [];
    const fromConfig = Object.keys(snapshot.model_configs ?? {});
    const fromEvents = Object.keys(eventsByModel);
    const fromReplicas = Object.keys(modelReplicas);
    const all = new Set([...fromConfig, ...fromEvents, ...fromReplicas]);
    return [...all].sort();
  }, [snapshot, eventsByModel, modelReplicas]);

  const launchFailuresByModel = useMemo(() => {
    const set = new Set<string>();
    if (!snapshot) return set;
    const cutoff = (snapshot.timestamp || 0) - 300;
    for (const f of snapshot.launch_failures ?? []) {
      if ((f.ts ?? 0) >= cutoff) set.add(f.model);
    }
    return set;
  }, [snapshot?.launch_failures, snapshot?.timestamp]);

  const ups = scaleEvents.filter((e) => e.action === "SCALE_UP").length;
  const downs = scaleEvents.filter((e) => e.action === "SCALE_DOWN").length;
  const zeros = scaleEvents.filter((e) => e.action === "SCALE_TO_ZERO").length;

  if (!snapshot) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-8 h-8 rounded-full border-2 border-cs-accent border-t-transparent animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      <TimeRangeBar
        preset={preset}
        onPreset={handlePreset}
        customStart={customStart}
        customEnd={customEnd}
        onCustomStart={setCustomStart}
        onCustomEnd={setCustomEnd}
        onApplyCustom={() => applyCustomRange(customStart, customEnd)}
        customError={customError}
        showApplyCustom
      />
      <p className="text-[10px] text-cs-dim -mt-4">
        Replica counts are live; scale events and KPIs use the range above
        (refreshed every 15s).
      </p>

      {/* KPI */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          {
            label: "Scale Ups",
            value: ups,
            icon: TrendingUp,
            color: "text-cs-accent",
          },
          {
            label: "Scale Downs",
            value: downs,
            icon: TrendingDown,
            color: "text-cs-warn",
          },
          {
            label: "Scale to Zero",
            value: zeros,
            icon: Minus,
            color: "text-cs-danger",
          },
          {
            label: "Active Models",
            value: models.length,
            icon: TrendingUp,
            color: "text-cs-accent2",
          },
        ].map((kpi) => (
          <div key={kpi.label} className="kpi-card flex items-center gap-3">
            <kpi.icon className={`w-5 h-5 ${kpi.color}`} />
            <div>
              <div className="text-[9px] text-cs-dim uppercase tracking-[0.15em] font-semibold">
                {kpi.label}
              </div>
              <div className="text-xl font-bold font-mono">{kpi.value}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Per-model cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {models.map((m) => (
          <ModelScaleCard
            key={m}
            model={m}
            replicas={modelReplicas[m] || 0}
            events={eventsByModel[m] || []}
            hasLaunchFailures={launchFailuresByModel.has(m)}
            preset={preset}
          />
        ))}
      </div>

      {/* Event log table */}
      <div>
        <h2 className="section-title">Scale Event Log</h2>
        <div className="card overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-cs-border">
                <th className="table-head">Time</th>
                <th className="table-head">Model</th>
                <th className="table-head">Action</th>
                <th className="table-head">Replicas</th>
                <th className="table-head">Reasons</th>
              </tr>
            </thead>
            <tbody>
              {scaleLoading ? (
                <tr>
                  <td
                    colSpan={5}
                    className="table-cell text-center text-cs-dim py-8"
                  >
                    Loading events…
                  </td>
                </tr>
              ) : (
                scaleEvents.map((e, i) => (
                  <tr key={`${e.timestamp}-${i}`} className="table-row">
                    <td className="table-cell font-mono text-cs-dim">
                      {formatEventTime(e.timestamp, preset)}
                    </td>
                    <td className="table-cell font-medium">{e.model}</td>
                    <td className="table-cell">
                      <ActionBadge action={e.action} />
                    </td>
                    <td className="table-cell font-mono">
                      {e.from_replicas} → {e.to_replicas}
                    </td>
                    <td className="table-cell text-cs-dim truncate max-w-xs">
                      {e.reasons.join("; ")}
                    </td>
                  </tr>
                ))
              )}
              {!scaleLoading && scaleEvents.length === 0 && (
                <tr>
                  <td
                    colSpan={5}
                    className="table-cell text-center text-cs-dim py-8"
                  >
                    No scale events in this window
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
