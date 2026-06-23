/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        cs: {
          bg: "var(--cs-bg)",
          surface: "var(--cs-surface)",
          card: "var(--cs-card)",
          border: "var(--cs-border)",
          border2: "var(--cs-border2)",
          accent: "var(--cs-accent)",
          accent2: "var(--cs-accent2)",
          warn: "var(--cs-warn)",
          danger: "var(--cs-danger)",
          text: "var(--cs-text)",
          muted: "var(--cs-muted)",
          dim: "var(--cs-dim)",
          inset: "var(--cs-inset)",
          hover: "var(--cs-hover)",
        },
      },
      fontFamily: {
        sans: ['"Geist"', '"Inter"', "system-ui", "sans-serif"],
        mono: ['"Geist Mono"', '"JetBrains Mono"', "monospace"],
      },
      boxShadow: {
        glow: "var(--cs-shadow-glow)",
        "glow-sm": "var(--cs-shadow-glow-sm)",
        "glow-accent": "var(--cs-shadow-glow-accent)",
        "inner-glow": "var(--cs-shadow-inner)",
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        noise: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E\")",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.4s ease-out",
        "glow-pulse": "glowPulse 2s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        glowPulse: {
          "0%, 100%": { boxShadow: "var(--cs-shadow-glow)" },
          "50%": { boxShadow: "var(--cs-shadow-glow-accent)" },
        },
      },
    },
  },
  plugins: [],
};
