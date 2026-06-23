import { Routes, Route, NavLink } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Bell,
  FlaskConical,
  Users,
  TrendingUp,
  Settings,
} from "lucide-react";

const COSMOS_LOGO_URL =
  "https://cosmos.ualr.edu/wp-content/uploads/2024/07/cropped-cropped-cosmos_trans_black_full-1-1.png";
import { useCluster } from "./hooks/useCluster";
import Topology from "./pages/Topology";
import Autoscaling from "./pages/Autoscaling";
import UsagePage from "./pages/Usage";
import UserManagement from "./pages/UserManagement";
import AdminControls from "./pages/AdminControls";
import Playground from "./pages/Playground";
import Notifications from "./pages/Notifications";
import ModelPage from "./pages/ModelPage";
import ThemeToggle from "./components/ThemeToggle";

const navItems = [
  { to: "/", icon: Activity, label: "Cluster", end: true },
  { to: "/playground", icon: FlaskConical, label: "Playground" },
  { to: "/autoscaling", icon: TrendingUp, label: "Scaling" },
  { to: "/usage", icon: BarChart3, label: "Usage" },
  { to: "/notifications", icon: Bell, label: "Alerts" },
  { to: "/users", icon: Users, label: "Keys" },
  { to: "/admin", icon: Settings, label: "Admin" },
];

export default function App() {
  const { snapshot, connected } = useCluster();
  const computeAlertCount = snapshot?.gpu_compute_notifications?.length ?? 0;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-cs-border bg-cs-bg/90 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-6 h-14 flex items-center gap-8">
          {/* Brand */}
          <div className="flex items-center gap-2.5 mr-2 shrink-0">
            <img
              src={COSMOS_LOGO_URL}
              alt="COSMOS"
              className="h-6 w-auto object-contain dark:invert"
              width={120}
              height={24}
              decoding="async"
            />
            <span className="text-[15px] font-semibold tracking-tight text-cs-text">
              CServe
            </span>
          </div>

          {/* Nav */}
          <nav className="flex gap-0.5">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-all duration-150 ${
                    isActive
                      ? "bg-cs-accent/10 text-cs-accent shadow-glow-sm"
                      : "text-cs-muted hover:text-cs-text hover:bg-cs-hover"
                  }`
                }
              >
                <item.icon className="w-3.5 h-3.5" />
                {item.label}
                {item.to === "/notifications" && computeAlertCount > 0 ? (
                  <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 rounded-full bg-cs-warn text-[10px] font-bold text-black flex items-center justify-center tabular-nums">
                    {computeAlertCount > 9 ? "9+" : computeAlertCount}
                  </span>
                ) : null}
              </NavLink>
            ))}
          </nav>

          {/* Theme + stats */}
          <div className="ml-auto flex items-center gap-4 text-[11px] font-mono">
            <ThemeToggle />
            {snapshot && (
              <>
                <div className="flex items-center gap-1.5">
                  <span className="text-cs-dim">Replicas</span>
                  <span className="text-cs-accent font-semibold">
                    {snapshot.stats.ready_replicas}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-cs-dim">Inflight</span>
                  <span className="text-cs-text font-semibold">
                    {snapshot.stats.total_inflight}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-cs-dim">GPUs</span>
                  <span className="text-cs-text font-semibold">
                    {snapshot.stats.free_gpus}
                    <span className="text-cs-dim">/{snapshot.stats.total_gpus}</span>
                  </span>
                </div>
              </>
            )}
            <div className="flex items-center gap-1.5">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  connected
                    ? "bg-cs-accent shadow-[0_0_6px_rgba(62,207,142,0.5)]"
                    : "bg-cs-danger animate-pulse"
                }`}
              />
              <span className={connected ? "text-cs-dim" : "text-cs-danger"}>
                {connected ? "Live" : "Disconnected"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-[1920px] mx-auto w-full px-6 py-5">
        <Routes>
          <Route path="/" element={<Topology snapshot={snapshot} />} />
          <Route
            path="/models/:modelName"
            element={<ModelPage snapshot={snapshot} />}
          />
          <Route path="/playground" element={<Playground snapshot={snapshot} />} />
          <Route
            path="/autoscaling"
            element={<Autoscaling snapshot={snapshot} />}
          />
          <Route path="/usage" element={<UsagePage />} />
          <Route
            path="/notifications"
            element={<Notifications snapshot={snapshot} />}
          />
          <Route path="/users" element={<UserManagement />} />
          <Route path="/admin" element={<AdminControls />} />
        </Routes>
      </main>
    </div>
  );
}
