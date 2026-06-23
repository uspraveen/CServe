/** Apply saved theme before React paint (avoids flash). Default: light. */
const STORAGE_KEY = "cserve-theme";

export type Theme = "dark" | "light";

export function getStoredTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return "light";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.dataset.theme = theme;
}

applyTheme(getStoredTheme());

export { STORAGE_KEY };
