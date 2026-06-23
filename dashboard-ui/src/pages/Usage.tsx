import { useCallback, useEffect, useState } from "react";
import { BarChart3, Clock, Cpu, Zap } from "lucide-react";
import TimeRangeBar from "../components/TimeRangeBar";
import {
  type TimeRangePreset,
  defaultCustomInputs,
  localInputToUnix,
  presetWindowS,
} from "../lib/timeRange";

interface UsageRow {
  user_id: string;
  model: string;
  requests: number;
  total_gpu_time_s: number;
  avg_latency_s: number;
  prompt_tokens: number;
  completion_tokens: number;
}

interface CustomEpoch {
  since: number;
  until: number;
}

export default function UsagePage() {
  const [usage, setUsage] = useState<UsageRow[]>([]);
  const [preset, setPreset] = useState<TimeRangePreset>("24h");
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const [customEpoch, setCustomEpoch] = useState<CustomEpoch | null>(null);
  const [customError, setCustomError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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

  const buildQuery = useCallback((): string | null => {
    if (preset === "custom") {
      if (!customEpoch) return null;
      return `since_ts=${customEpoch.since}&until_ts=${customEpoch.until}`;
    }
    return `window_s=${presetWindowS(preset)}`;
  }, [preset, customEpoch]);

  const fetchUsage = useCallback(
    (showLoading = false) => {
      const q = buildQuery();
      if (!q) {
        setLoading(false);
        return;
      }
      if (showLoading) setLoading(true);
      fetch(`/dashboard/api/usage?${q}`)
        .then((r) => r.json())
        .then((data) => {
          setUsage(data);
          setLoading(false);
        })
        .catch(() => setLoading(false));
    },
    [buildQuery],
  );

  useEffect(() => {
    fetchUsage(true);
  }, [fetchUsage]);

  useEffect(() => {
    const interval = setInterval(() => fetchUsage(false), 5000);
    return () => clearInterval(interval);
  }, [fetchUsage]);

  const totalRequests = usage.reduce((s, r) => s + r.requests, 0);
  const totalGpuTime = usage.reduce((s, r) => s + r.total_gpu_time_s, 0);
  const totalTokens = usage.reduce(
    (s, r) => s + r.prompt_tokens + r.completion_tokens,
    0,
  );
  const uniqueUsers = new Set(usage.map((r) => r.user_id)).size;

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

      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          {
            label: "Total Requests",
            value: totalRequests.toLocaleString(),
            icon: BarChart3,
            color: "text-cs-accent",
          },
          {
            label: "GPU Time",
            value: `${totalGpuTime.toFixed(1)}s`,
            icon: Cpu,
            color: "text-cs-accent2",
          },
          {
            label: "Total Tokens",
            value: totalTokens.toLocaleString(),
            icon: Zap,
            color: "text-cs-warn",
          },
          {
            label: "Active Users",
            value: uniqueUsers,
            icon: Clock,
            color: "text-blue-400",
          },
        ].map((kpi) => (
          <div key={kpi.label} className="kpi-card">
            <div className="flex items-center gap-2 mb-1">
              <kpi.icon className={`w-4 h-4 ${kpi.color}`} />
              <div className="text-[9px] text-cs-dim uppercase tracking-[0.15em] font-semibold">
                {kpi.label}
              </div>
            </div>
            <div className="text-2xl font-bold font-mono">{kpi.value}</div>
          </div>
        ))}
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-cs-border">
              <th className="table-head">User</th>
              <th className="table-head">Model</th>
              <th className="table-head text-right">Requests</th>
              <th className="table-head text-right">GPU Time</th>
              <th className="table-head text-right">Avg Latency</th>
              <th className="table-head text-right">Prompt Tokens</th>
              <th className="table-head text-right">Completion Tokens</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td
                  colSpan={7}
                  className="table-cell text-center text-cs-dim py-8"
                >
                  Loading...
                </td>
              </tr>
            ) : usage.length === 0 ? (
              <tr>
                <td
                  colSpan={7}
                  className="table-cell text-center text-cs-dim py-8"
                >
                  No usage data in this window
                </td>
              </tr>
            ) : (
              usage.map((row, i) => (
                <tr
                  key={`${row.user_id}-${row.model}-${i}`}
                  className="table-row"
                >
                  <td className="table-cell font-medium">{row.user_id}</td>
                  <td className="table-cell text-cs-muted">{row.model}</td>
                  <td className="table-cell text-right font-mono">
                    {row.requests.toLocaleString()}
                  </td>
                  <td className="table-cell text-right font-mono">
                    {row.total_gpu_time_s.toFixed(1)}s
                  </td>
                  <td className="table-cell text-right font-mono">
                    {(row.avg_latency_s * 1000).toFixed(0)}ms
                  </td>
                  <td className="table-cell text-right font-mono">
                    {row.prompt_tokens.toLocaleString()}
                  </td>
                  <td className="table-cell text-right font-mono">
                    {row.completion_tokens.toLocaleString()}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
