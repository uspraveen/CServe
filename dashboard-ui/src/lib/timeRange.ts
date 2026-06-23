export type TimeRangePreset = "24h" | "7d" | "30d" | "custom";

export const SEC_24H = 86400;
export const SEC_7D = 604800;
/** Rolling 30-day window (~1 month). */
export const SEC_30D = 86400 * 30;

export function toLocalInput(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function localInputToUnix(s: string): number {
  const t = new Date(s).getTime();
  return Number.isNaN(t) ? NaN : t / 1000;
}

export function defaultCustomInputs(): { start: string; end: string } {
  const end = new Date();
  const start = new Date(end.getTime() - SEC_24H * 1000);
  return { start: toLocalInput(start), end: toLocalInput(end) };
}

export function presetWindowS(p: Exclude<TimeRangePreset, "custom">): number {
  switch (p) {
    case "24h":
      return SEC_24H;
    case "7d":
      return SEC_7D;
    case "30d":
      return SEC_30D;
  }
}
