import { useEffect, useState, useCallback } from "react";
import { Plus, Trash2, EyeOff, Shield, User } from "lucide-react";

interface ApiKeyInfo {
  key_id: string;
  key_prefix: string;
  user_id: string;
  name: string;
  role: string;
  rate_limit_rpm: number;
  enabled: boolean;
  created_at: number;
  last_used_at: number;
  total_requests: number;
}

interface UserInfo {
  user_id: string;
  key_count: number;
  total_requests: number;
  last_active: number;
  first_key_at: number;
}

function RoleBadge({ role }: { role: string }) {
  return role === "admin" ? (
    <span className="badge bg-cs-accent2/10 text-cs-accent2 border border-cs-accent2/20 flex items-center gap-0.5 w-fit">
      <Shield className="w-3 h-3" />
      ADMIN
    </span>
  ) : (
    <span className="badge bg-cs-border text-cs-dim border border-cs-border2 flex items-center gap-0.5 w-fit">
      <User className="w-3 h-3" />
      USER
    </span>
  );
}

export default function UserManagement() {
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({
    user_id: "",
    name: "",
    role: "user",
    rate_limit_rpm: 0,
  });

  const refresh = useCallback(() => {
    fetch("/dashboard/api/users")
      .then((r) => r.json())
      .then(setUsers)
      .catch(() => {});
    fetch("/dashboard/api/keys")
      .then((r) => r.json())
      .then(setKeys)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleCreate = async () => {
    if (!form.user_id) return;
    const adminKey = prompt("Enter your admin API key (csk_...):");
    if (!adminKey) return;

    const resp = await fetch("/admin/keys", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${adminKey}`,
      },
      body: JSON.stringify(form),
    });
    const data = await resp.json();
    if (data.key) {
      setNewKey(data.key);
      setShowForm(false);
      setForm({ user_id: "", name: "", role: "user", rate_limit_rpm: 0 });
      refresh();
    } else {
      alert(data.error?.message || "Failed to create key");
    }
  };

  const handleRevoke = async (keyId: string) => {
    if (!confirm("Revoke this API key?")) return;
    const adminKey = prompt("Enter your admin API key:");
    if (!adminKey) return;

    await fetch(`/admin/keys/${keyId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${adminKey}` },
    });
    refresh();
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* New key reveal */}
      {newKey && (
        <div className="card border-cs-accent/20 bg-cs-accent/[0.03] p-5">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold text-cs-accent mb-1">
                New API Key Created
              </div>
              <div className="text-[11px] text-cs-dim mb-3">
                Copy this key now — it won&apos;t be shown again.
              </div>
              <code className="block bg-cs-surface rounded-lg px-4 py-2.5 font-mono text-sm select-all border border-cs-border">
                {newKey}
              </code>
            </div>
            <button
              onClick={() => setNewKey(null)}
              className="text-cs-dim hover:text-cs-text transition-colors"
            >
              <EyeOff className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      {/* Users */}
      <div>
        <h2 className="section-title">Users</h2>
        <div className="card overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-cs-border">
                <th className="table-head">User ID</th>
                <th className="table-head text-right">API Keys</th>
                <th className="table-head text-right">Total Requests</th>
                <th className="table-head text-right">Last Active</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.user_id} className="table-row">
                  <td className="table-cell font-medium">{u.user_id}</td>
                  <td className="table-cell text-right font-mono">
                    {u.key_count}
                  </td>
                  <td className="table-cell text-right font-mono">
                    {u.total_requests.toLocaleString()}
                  </td>
                  <td className="table-cell text-right text-cs-dim font-mono">
                    {u.last_active > 0
                      ? new Date(u.last_active * 1000).toLocaleString()
                      : "Never"}
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr>
                  <td
                    colSpan={4}
                    className="table-cell text-center text-cs-dim py-8"
                  >
                    No users yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* API Keys */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="section-title mb-0">API Keys</h2>
          <button
            onClick={() => setShowForm(!showForm)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-cs-accent text-black hover:bg-cs-accent/90 transition-all shadow-glow-sm"
          >
            <Plus className="w-3.5 h-3.5" /> New Key
          </button>
        </div>

        {/* Form */}
        {showForm && (
          <div className="card p-5 mb-4 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] text-cs-dim font-semibold uppercase tracking-wider mb-1.5">
                  User ID
                </label>
                <input
                  value={form.user_id}
                  onChange={(e) =>
                    setForm({ ...form, user_id: e.target.value })
                  }
                  className="w-full bg-cs-surface border border-cs-border rounded-lg px-3 py-2 text-sm font-mono text-cs-text placeholder:text-cs-dim/50 focus:outline-none focus:border-cs-accent/30 focus:shadow-glow-sm transition-all"
                  placeholder="e.g. alice"
                />
              </div>
              <div>
                <label className="block text-[10px] text-cs-dim font-semibold uppercase tracking-wider mb-1.5">
                  Key Name
                </label>
                <input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  className="w-full bg-cs-surface border border-cs-border rounded-lg px-3 py-2 text-sm font-mono text-cs-text placeholder:text-cs-dim/50 focus:outline-none focus:border-cs-accent/30 focus:shadow-glow-sm transition-all"
                  placeholder="e.g. production-backend"
                />
              </div>
              <div>
                <label className="block text-[10px] text-cs-dim font-semibold uppercase tracking-wider mb-1.5">
                  Role
                </label>
                <select
                  value={form.role}
                  onChange={(e) => setForm({ ...form, role: e.target.value })}
                  className="w-full bg-cs-surface border border-cs-border rounded-lg px-3 py-2 text-sm text-cs-text focus:outline-none focus:border-cs-accent/30 transition-all"
                >
                  <option value="user">User</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
              <div>
                <label className="block text-[10px] text-cs-dim font-semibold uppercase tracking-wider mb-1.5">
                  Rate Limit (req/min, 0=∞)
                </label>
                <input
                  type="number"
                  value={form.rate_limit_rpm}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      rate_limit_rpm: parseInt(e.target.value) || 0,
                    })
                  }
                  className="w-full bg-cs-surface border border-cs-border rounded-lg px-3 py-2 text-sm font-mono text-cs-text focus:outline-none focus:border-cs-accent/30 focus:shadow-glow-sm transition-all"
                />
              </div>
            </div>
            <button
              onClick={handleCreate}
              className="px-5 py-2 rounded-lg text-[11px] font-semibold bg-cs-accent text-black hover:bg-cs-accent/90 transition-all shadow-glow-sm"
            >
              Create API Key
            </button>
          </div>
        )}

        {/* Keys table */}
        <div className="card overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-cs-border">
                <th className="table-head">Key</th>
                <th className="table-head">User</th>
                <th className="table-head">Name</th>
                <th className="table-head">Role</th>
                <th className="table-head text-right">Rate Limit</th>
                <th className="table-head text-right">Requests</th>
                <th className="table-head">Status</th>
                <th className="table-head text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr
                  key={k.key_id}
                  className={`table-row ${!k.enabled ? "opacity-40" : ""}`}
                >
                  <td className="table-cell font-mono text-cs-dim">
                    {k.key_prefix}...
                  </td>
                  <td className="table-cell font-medium">{k.user_id}</td>
                  <td className="table-cell text-cs-muted">{k.name}</td>
                  <td className="table-cell">
                    <RoleBadge role={k.role} />
                  </td>
                  <td className="table-cell text-right font-mono">
                    {k.rate_limit_rpm > 0 ? `${k.rate_limit_rpm}/min` : "∞"}
                  </td>
                  <td className="table-cell text-right font-mono">
                    {k.total_requests.toLocaleString()}
                  </td>
                  <td className="table-cell">
                    {k.enabled ? (
                      <span className="badge bg-cs-accent/10 text-cs-accent border border-cs-accent/20">
                        ACTIVE
                      </span>
                    ) : (
                      <span className="badge bg-cs-danger/10 text-cs-danger border border-cs-danger/20">
                        REVOKED
                      </span>
                    )}
                  </td>
                  <td className="table-cell text-right">
                    {k.enabled && (
                      <button
                        onClick={() => handleRevoke(k.key_id)}
                        className="text-cs-dim hover:text-cs-danger transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {keys.length === 0 && (
                <tr>
                  <td
                    colSpan={8}
                    className="table-cell text-center text-cs-dim py-8"
                  >
                    No API keys yet
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
