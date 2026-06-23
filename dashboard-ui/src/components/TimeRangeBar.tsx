import { CalendarRange } from "lucide-react";
import type { TimeRangePreset } from "../lib/timeRange";

const PRESETS: { id: TimeRangePreset; label: string }[] = [
  { id: "24h", label: "24h" },
  { id: "7d", label: "7 days" },
  { id: "30d", label: "30 days" },
  { id: "custom", label: "Custom" },
];

interface Props {
  preset: TimeRangePreset;
  onPreset: (p: TimeRangePreset) => void;
  customStart: string;
  customEnd: string;
  onCustomStart: (v: string) => void;
  onCustomEnd: (v: string) => void;
  onApplyCustom?: () => void;
  customError?: string | null;
  /** When true, custom row shows an Apply control (Usage uses polling; Scaling refetches on apply). */
  showApplyCustom?: boolean;
}

export default function TimeRangeBar({
  preset,
  onPreset,
  customStart,
  customEnd,
  onCustomStart,
  onCustomEnd,
  onApplyCustom,
  customError,
  showApplyCustom,
}: Props) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[10px] text-cs-dim uppercase tracking-[0.12em] font-semibold mr-1 flex items-center gap-1">
          <CalendarRange className="w-3 h-3" />
          Range
        </span>
        {PRESETS.map((opt) => (
          <button
            key={opt.id}
            type="button"
            onClick={() => onPreset(opt.id)}
            className={`px-3.5 py-1.5 rounded-lg text-[11px] font-semibold transition-all duration-150 ${
              preset === opt.id
                ? "bg-cs-accent/10 text-cs-accent border border-cs-accent/20 shadow-glow-sm"
                : "bg-cs-card text-cs-dim border border-cs-border hover:text-cs-text hover:border-cs-border2"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {preset === "custom" && (
        <div className="flex flex-wrap items-end gap-3 p-3 rounded-lg border border-cs-border bg-cs-card/40">
          <label className="flex flex-col gap-1 text-[10px] text-cs-dim uppercase tracking-wide font-semibold">
            From
            <input
              type="datetime-local"
              value={customStart}
              onChange={(e) => onCustomStart(e.target.value)}
              className="px-2 py-1.5 rounded-md bg-cs-bg border border-cs-border text-cs-text text-xs font-mono"
            />
          </label>
          <label className="flex flex-col gap-1 text-[10px] text-cs-dim uppercase tracking-wide font-semibold">
            To
            <input
              type="datetime-local"
              value={customEnd}
              onChange={(e) => onCustomEnd(e.target.value)}
              className="px-2 py-1.5 rounded-md bg-cs-bg border border-cs-border text-cs-text text-xs font-mono"
            />
          </label>
          {showApplyCustom && onApplyCustom && (
            <button
              type="button"
              onClick={onApplyCustom}
              className="px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-cs-accent/15 text-cs-accent border border-cs-accent/25 hover:bg-cs-accent/25"
            >
              Apply range
            </button>
          )}
          {customError && (
            <span className="text-[11px] text-cs-danger w-full">{customError}</span>
          )}
          <span className="text-[10px] text-cs-dim w-full">
            Times use your browser&apos;s local timezone
            {showApplyCustom ? " · click Apply after editing" : ""}
          </span>
        </div>
      )}
    </div>
  );
}
