import { Moon, Sun } from "lucide-react";
import { useTheme } from "../hooks/useTheme";

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-cs-border bg-cs-surface/80 text-cs-muted hover:text-cs-accent hover:border-cs-accent/30 transition-colors"
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
    >
      {isDark ? (
        <Sun className="w-3.5 h-3.5 text-cs-accent" />
      ) : (
        <Moon className="w-3.5 h-3.5 text-cs-accent" />
      )}
      <span className="text-[11px] font-medium hidden sm:inline">
        {isDark ? "Light" : "Dark"}
      </span>
    </button>
  );
}
