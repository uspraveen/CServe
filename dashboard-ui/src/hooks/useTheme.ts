import { useCallback, useSyncExternalStore } from "react";
import {
  applyTheme,
  getStoredTheme,
  STORAGE_KEY,
  type Theme,
} from "../theme-init";

function subscribe(onStoreChange: () => void) {
  const handler = () => onStoreChange();
  window.addEventListener("cserve-theme-change", handler);
  return () => window.removeEventListener("cserve-theme-change", handler);
}

function getSnapshot(): Theme {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, () => "light");

  const setTheme = useCallback((next: Theme) => {
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
    window.dispatchEvent(new Event("cserve-theme-change"));
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  return { theme, setTheme, toggleTheme, isDark: theme === "dark" };
}

export function initThemeFromStorage(): Theme {
  const t = getStoredTheme();
  applyTheme(t);
  return t;
}
