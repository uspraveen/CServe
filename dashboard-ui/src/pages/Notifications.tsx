import { useEffect, useReducer, useMemo } from "react";
import { Bell, Cpu, Gauge, Radio } from "lucide-react";
import type { ClusterSnapshot, GpuComputeNotification } from "../hooks/useCluster";

function liveSustainProgress(
  snapshot: ClusterSnapshot,
  row: GpuComputeNotification,
): { elapsed: number; progress: number } {
  const duration = Math.max(0.001, row.sustain_duration_s);
  const drift = Math.max(0, Date.now() / 1000 - snapshot.timestamp);
  const elapsed = Math.min(duration, row.sustain_elapsed_s + drift);
  return { elapsed, progress: Math.min(1, elapsed / duration) };
}

function NotificationCard({
  snapshot,
  row,
}: {
  snapshot: ClusterSnapshot;
  row: GpuComputeNotification;
}) {
  const [, tick] = useReducer((n) => n + 1, 0);

  useEffect(() => {
    const id = window.setInterval(() => tick(), 80);
    return () => window.clearInterval(id);
  }, [snapshot.timestamp, row.sustain_elapsed_s, row.sustain_duration_s]);

  const { elapsed, progress } = liveSustainProgress(snapshot, row);
  const utilPct = Math.round(row.compute_util_frac * 100);
  const threshPct = Math.round(row.threshold_frac * 100);
  const trip = progress >= 1;
  const duration = row.sustain_duration_s;

  return (
    <article
      className="relative overflow-hidden rounded-xl border border-cs-border bg-cs-card/80 backdrop-blur-sm"
      style={{
        boxShadow: trip
          ? "inset 0 0 0 1px rgba(242, 78, 78, 0.25), 0 0 40px rgba(245, 166, 35, 0.06)"
          : "inset 0 0 0 1px rgba(245, 166, 35, 0.12)",
      }}
    >
      <div
        className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-cs-warn via-amber-500/80 to-orange-600/60"
        aria-hidden
      />
      <div className="pl-5 pr-4 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-[11px] font-mono text-cs-dim uppercase tracking-widest">
              <Radio
                className={`w-3.5 h-3.5 ${trip ? "text-cs-danger animate-pulse" : "text-cs-warn"}`}
              />
              GPU compute pressure
            </div>
            <h3 className="mt-1.5 font-['Syne',system-ui,sans-serif] text-lg font-semibold tracking-tight text-cs-text">
              {row.node_name}
              <span className="text-cs-dim font-normal font-mono text-sm ml-2">
                · GPU {row.gpu_index}
              </span>
            </h3>
            {row.replica_id ? (
              <p className="mt-1 text-[11px] font-mono text-cs-muted truncate max-w-md">
                replica {row.replica_id}
              </p>
            ) : null}
          </div>
          <div className="flex flex-col items-end gap-1">
            <span
              className={`badge ${
                trip
                  ? "bg-cs-danger/15 text-red-300 border border-cs-danger/25"
                  : "bg-cs-warn/10 text-amber-200 border border-cs-warn/20"
              }`}
            >
              {row.guard_state}
            </span>
            <div className="flex flex-col items-end gap-0.5 font-mono text-[10px] text-right">
              <div>
                <span className="text-cs-dim">GPU compute now </span>
                <span className="text-cs-warn font-semibold tabular-nums">{utilPct}%</span>
                <span className="text-cs-dim"> (nvidia-smi util, not VRAM)</span>
              </div>
              <div>
                <span className="text-cs-dim">Policy threshold (compute) </span>
                <span className="text-cs-muted font-semibold tabular-nums">{threshPct}%</span>
                <span className="text-cs-dim"> · cluster safety</span>
              </div>
            </div>
          </div>
        </div>

        <p className="mt-4 text-[12px] text-cs-muted leading-relaxed max-w-2xl">
          {trip
            ? "Sustained high GPU compute utilization (SM busy %) — the guard treats this like VRAM pressure (warn → mitigate → migrate). This is not the same as VRAM used / total."
            : `GPU compute utilization has been at or above ${threshPct}% (policy threshold). If that holds continuously for ${duration.toFixed(0)}s, the guard escalates. Same metric as nvidia-smi GPU-Util column — not Memory-Usage.`}
        </p>

        <div className="mt-5">
          <div className="flex justify-between items-center text-[10px] font-mono uppercase tracking-wider text-cs-dim mb-2">
            <span className="flex items-center gap-1.5">
              <Gauge className="w-3 h-3" />
              Sustain timer
            </span>
            <span className="tabular-nums text-cs-text">
              {elapsed.toFixed(1)}s / {duration.toFixed(0)}s
            </span>
          </div>
          {/* Slider-style track: read-only visual (thumb follows server + local drift) */}
          <div
            className="relative h-4 rounded-full bg-cs-border/80 overflow-visible"
            role="group"
            aria-label={`Sustained utilization timer, ${elapsed.toFixed(0)} of ${duration.toFixed(0)} seconds`}
          >
            <div
              className={`absolute inset-y-0 left-0 rounded-full transition-[width] duration-75 ${
                trip
                  ? "bg-gradient-to-r from-cs-danger/90 to-red-600/80"
                  : "bg-gradient-to-r from-amber-600/50 via-cs-warn/70 to-amber-400/40"
              }`}
              style={{ width: `${progress * 100}%` }}
            />
            <div
              className="absolute top-1/2 h-5 w-5 -mt-2.5 rounded-full border-2 border-cs-text/90 bg-cs-card shadow-[0_0_12px_rgba(245,166,35,0.35)] pointer-events-none transition-[left] duration-75"
              style={{
                left: `clamp(0px, calc(${progress * 100}% - 10px), calc(100% - 20px))`,
              }}
            />
          </div>
        </div>
      </div>
    </article>
  );
}

export default function Notifications({
  snapshot,
}: {
  snapshot: ClusterSnapshot | null;
}) {
  const rows = snapshot?.gpu_compute_notifications ?? [];
  const threshold =
    snapshot?.gpu_guard?.compute_sustain_threshold ?? 0.95;
  const durationS =
    snapshot?.gpu_guard?.compute_sustain_duration_s ?? 900;

  const sorted = useMemo(
    () => [...rows].sort((a, b) => b.sustain_progress - a.sustain_progress),
    [rows],
  );

  return (
    <div className="max-w-3xl mx-auto pb-16">
      <div className="mb-8 relative">
        <div
          className="absolute -inset-8 opacity-[0.07] pointer-events-none bg-noise bg-[length:200px_200px] rounded-3xl"
          aria-hidden
        />
        <div className="relative flex items-start gap-4">
          <div className="p-3 rounded-2xl bg-cs-warn/10 border border-cs-warn/15">
            <Bell className="w-7 h-7 text-cs-warn" strokeWidth={1.5} />
          </div>
          <div>
            <h1 className="font-['Syne',system-ui,sans-serif] text-3xl font-bold tracking-tight text-cs-text">
              Notifications
            </h1>
            <p className="mt-2 text-sm text-cs-muted leading-relaxed max-w-xl">
              Alerts when GPU compute utilization (nvidia-smi) stays at or above{" "}
              <span className="text-cs-warn font-mono tabular-nums">
                {(threshold * 100).toFixed(0)}%
              </span>
              . Each card shows a live sustain timer toward the guard threshold (
              <span className="font-mono tabular-nums">{durationS.toFixed(0)}s</span>{" "}
              continuous by default).
            </p>
          </div>
        </div>
      </div>

      {!snapshot ? (
        <div className="card p-10 text-center text-cs-muted text-sm">
          Waiting for cluster snapshot…
        </div>
      ) : sorted.length === 0 ? (
        <div className="card p-12 text-center border-dashed border-cs-border2">
          <Cpu className="w-10 h-10 mx-auto text-cs-dim opacity-40 mb-4" />
          <p className="text-cs-text font-medium">No GPU compute alerts</p>
          <p className="text-sm text-cs-muted mt-2 max-w-md mx-auto">
            All online GPUs are below the sustained compute threshold (
            {(threshold * 100).toFixed(0)}%). Alerts appear here when any GPU
            crosses that level so you can watch the {durationS.toFixed(0)}s
            countdown before the guard acts.
          </p>
        </div>
      ) : (
        <ul className="space-y-5">
          {sorted.map((row) => (
            <li key={`${row.node_name}-${row.gpu_index}`}>
              <NotificationCard snapshot={snapshot} row={row} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
