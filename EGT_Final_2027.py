"""
Corrected evolutionary game model for PQC migration in CII.
Includes:
- Incoming weight normalization.
- Mutation (spontaneous strategy switching) in agent-based model.
- Ensemble averaging for stochastic simulations.
- Replicator equation with mutation term.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Parameters
# ===============================
params = {
    'N': 100,                # number of nodes
    'T_Q': 10.0,             # Q-Day horizon
    'lambda_risk': 0.5,      # quantum progress speed
    'v': 1.0,                # value of long-lived info
    'alpha_bar': 0.8,        # average vulnerability
    'beta': 0.3,             # security bonus coefficient
    'C_bar': 0.2,            # avg migration cost (for replicator)
    'c_base': 0.1,           # fixed integration cost
    'c_frict': 0.5,          # technological friction coefficient
    'gamma': 2.0,            # nonlinear degradation exponent
    'kappa': 5.0,            # selection intensity (Fermi)
    'mu': 1e-4,              # mutation/innovation probability (per update)
    'x0': 0.05,              # initial fraction of PQC adopters
    't_max': 20.0,           # simulation time
    'dt_agent': 0.1,         # time step for agent-based simulation
    'ensemble_runs': 20,     # number of realizations for averaging
}

# ===============================
# 2. Graph creation with incoming weight normalization
# ===============================
def create_graph(topology='complete', N=None):
    if N is None:
        N = params['N']
    if topology == 'complete':
        G = nx.complete_graph(N).to_directed()
    elif topology == 'erdos_renyi':
        G = nx.erdos_renyi_graph(N, 0.1, directed=True)
    elif topology == 'barabasi_albert':
        G = nx.barabasi_albert_graph(N, 2).to_directed()
    else:
        G = nx.erdos_renyi_graph(N, 0.2, directed=True)
    # Random weights uniform (0.1,1]
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.1, 1.0)
    # Normalize incoming edges per node (sum of incoming weights = 1)
    for node in G.nodes():
        in_edges = list(G.in_edges(node))
        total = sum(G[u][node]['weight'] for u, node in in_edges)
        if total > 0:
            for u, _ in in_edges:
                G[u][node]['weight'] /= total
    return G

# ===============================
# 3. Costs based on topological in-degree
# ===============================
def compute_costs(G, c_base, c_frict, gamma):
    in_deg = dict(G.in_degree())
    k_max = max(in_deg.values()) if in_deg else 1
    costs = {}
    for i in G.nodes():
        k_in = in_deg[i]
        costs[i] = c_base + c_frict * (k_in / k_max) ** gamma
    return costs

# ===============================
# 4. Macroscopic replicator with mutation
# ===============================
def replicator_rhs(x, t, C_bar, alpha_bar, lam, T_Q, v, mu):
    R = v * np.exp(lam * (t - T_Q))
    G = alpha_bar * R * (1 - x) - C_bar
    return x * (1 - x) * G + mu * (1 - 2 * x)

def solve_replicator(C_bar, alpha_bar, lam, T_Q, v, t_span, x0, mu):
    t = np.linspace(t_span[0], t_span[1], 500)
    sol = odeint(replicator_rhs, x0, t, args=(C_bar, alpha_bar, lam, T_Q, v, mu))
    return t, sol.flatten()

def t_star_analytic(C_bar, alpha_bar, lam, T_Q, v, x_crit=0.5):
    arg = C_bar / (alpha_bar * v * (1 - x_crit))
    if arg <= 0:
        return np.inf
    return T_Q + (1/lam) * np.log(arg)

# ===============================
# 5. Agent-based simulation with mutation
# ===============================
def fitness_node(s, node, t, G, costs, alpha_bar, beta, v, lam, T_Q):
    U_net = 0.0
    for pred in G.predecessors(node):
        w = G[pred][node]['weight']
        U_net += beta * w * s[pred]
    R_val = v * np.exp(lam * (t - T_Q))
    sum_w_s = 0.0
    for pred in G.predecessors(node):
        w = G[pred][node]['weight']
        sum_w_s += w * s[pred]
    unprotected = max(0.0, 1.0 - sum_w_s)
    if s[node] == 1:
        return U_net - costs[node]
    else:
        return U_net - alpha_bar * R_val * unprotected

def fermi_prob(delta_f, kappa):
    return 1.0 / (1.0 + np.exp(-kappa * delta_f))

def agent_based_simulation(G, costs, T_Q, lam, v, alpha_bar, beta, kappa, mu,
                           t_max, dt, x0):
    N = G.number_of_nodes()
    # Initial strategies
    s = {i: 1 if np.random.rand() < x0 else 0 for i in G.nodes()}
    times = np.arange(0, t_max, dt)
    x_hist = []
    edges = list(G.edges())
    if len(edges) == 0:
        edges = [(i, i) for i in G.nodes()]
    for t in times:
        for _ in range(N):
            # Mutation (innovation) step with probability mu
            if np.random.rand() < mu:
                # Pick a random node and flip its strategy
                node = np.random.choice(list(G.nodes()))
                s[node] = 1 - s[node]
            else:
                # Standard Fermi copying
                i, j = edges[np.random.randint(len(edges))]
                fit_i = fitness_node(s, i, t, G, costs, alpha_bar, beta, v, lam, T_Q)
                fit_j = fitness_node(s, j, t, G, costs, alpha_bar, beta, v, lam, T_Q)
                prob = fermi_prob(fit_i - fit_j, kappa)
                if np.random.rand() < prob:
                    s[j] = s[i]
        x_hist.append(np.mean(list(s.values())))
    return times, np.array(x_hist)

# ===============================
# 6. Main experimental routine
# ===============================
def run_experiments():
    p = params

    # ---------- Figure 1: Replicator for different C_bar ----------
    plt.figure(figsize=(10,6))
    C_vals = [0.1, 0.2, 0.3]
    t_span = (0, p['t_max'])
    for C in C_vals:
        t, x = solve_replicator(C, p['alpha_bar'], p['lambda_risk'],
                                p['T_Q'], p['v'], t_span, p['x0'], p['mu'])
        plt.plot(t, x, lw=2.5, label=f'$\\overline{{C}} = {C}$')
        t_an = t_star_analytic(C, p['alpha_bar'], p['lambda_risk'], p['T_Q'], p['v'], 0.5)
        if np.isfinite(t_an):
            plt.axvline(t_an, linestyle='--', alpha=0.5, color=plt.gca().lines[-1].get_color())
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Fraction of PQC adopters $x(t)$', fontsize=12)
    plt.title('Replicator dynamics: effect of migration cost', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, p['t_max'])
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('fig1_replicator_C.png', dpi=300)
    plt.show()

    # ---------- Figure 2: Replicator for different lambda ----------
    plt.figure(figsize=(10,6))
    lam_vals = [0.3, 0.5, 0.7]
    for lam in lam_vals:
        t, x = solve_replicator(p['C_bar'], p['alpha_bar'], lam,
                                p['T_Q'], p['v'], t_span, p['x0'], p['mu'])
        plt.plot(t, x, lw=2.5, label=f'$\\lambda = {lam}$')
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('$x(t)$', fontsize=12)
    plt.title('Influence of quantum progress speed $\\lambda$', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_replicator_lambda.png', dpi=300)
    plt.show()

    # ---------- Figure 3: Analytical t* vs numerical (replicator) ----------
    C_range = np.linspace(0.05, 0.4, 20)
    t_num_vals = []
    t_an_vals = []
    for C in C_range:
        def ode_wrapper(t, x):
            return replicator_rhs(x, t, C, p['alpha_bar'], p['lambda_risk'],
                                  p['T_Q'], p['v'], p['mu'])
        sol = solve_ivp(ode_wrapper, (0, 2*p['T_Q']), [p['x0']], t_eval=np.linspace(0, 2*p['T_Q'], 500))
        x_vals = sol.y[0]
        t_vals = sol.t
        idx = np.where(x_vals >= 0.5)[0]
        if len(idx) > 0:
            t_num_vals.append(t_vals[idx[0]])
        else:
            t_num_vals.append(np.nan)
        t_an_vals.append(t_star_analytic(C, p['alpha_bar'], p['lambda_risk'], p['T_Q'], p['v'], 0.5))
    plt.figure(figsize=(8,6))
    plt.plot(C_range, t_an_vals, 'b-', lw=2.5, label='Analytical $t^*$ (bifurcation point)')
    plt.plot(C_range, t_num_vals, 'ro', markersize=4, label='Numerical $t^*$ (time to reach $x=0.5$)')
    plt.xlabel('Average migration cost $\\overline{C}$', fontsize=12)
    plt.ylabel('Tipping point $t^*$', fontsize=12)
    plt.title('Systemic inertia: gap between onset of cooperation and majority adoption', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_tstar_validation.png', dpi=300)
    plt.show()

    # ---------- Figure 4: Agent-based vs replicator (complete graph, ensemble average) ----------
    print("Figure 4: running agent-based ensemble on complete graph...")
    G_comp = create_graph('complete', N=50)
    costs_comp = compute_costs(G_comp, p['c_base'], p['c_frict'], p['gamma'])
    C_avg_comp = np.mean(list(costs_comp.values()))
    # Ensemble averaging
    x_ensemble = []
    t_ref = None
    for run in range(p['ensemble_runs']):
        t_agent, x_agent = agent_based_simulation(G_comp, costs_comp, p['T_Q'],
                                                  p['lambda_risk'], p['v'], p['alpha_bar'],
                                                  p['beta'], p['kappa'], p['mu'],
                                                  t_max=15.0, dt=p['dt_agent'], x0=p['x0'])
        if t_ref is None:
            t_ref = t_agent
        x_ensemble.append(x_agent)
    x_agent_mean = np.mean(x_ensemble, axis=0)
    # Replicator solution
    t_rep, x_rep = solve_replicator(C_avg_comp, p['alpha_bar'], p['lambda_risk'],
                                    p['T_Q'], p['v'], (0,15), p['x0'], p['mu'])
    plt.figure(figsize=(10,6))
    plt.plot(t_ref, x_agent_mean, 'b-', lw=2, label='Agent-based (ensemble average, complete graph)')
    plt.plot(t_rep, x_rep, 'r--', lw=2.5, label='Replicator equation (mean-field)')
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Fraction of PQC adopters $x(t)$', fontsize=12)
    plt.title('Microscopic vs macroscopic dynamics (with mutation)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig4_agent_vs_replicator.png', dpi=300)
    plt.show()

    # ---------- Figure 5: Heatmap of t*(C, lambda) ----------
    C_heat = np.linspace(0.05, 0.4, 30)
    lam_heat = np.linspace(0.2, 1.0, 30)
    tstar_mat = np.zeros((len(lam_heat), len(C_heat)))
    for i, lam in enumerate(lam_heat):
        for j, C in enumerate(C_heat):
            ts = t_star_analytic(C, p['alpha_bar'], lam, p['T_Q'], p['v'], x_crit=0.5)
            tstar_mat[i,j] = ts if ts < 30 else 30
    plt.figure(figsize=(10,8))
    im = plt.imshow(tstar_mat, extent=[C_heat[0], C_heat[-1], lam_heat[-1], lam_heat[0]],
                    aspect='auto', cmap='viridis', origin='upper')
    plt.colorbar(im, label='$t^*$')
    plt.xlabel('Average cost $\\overline{C}$', fontsize=12)
    plt.ylabel('Quantum progress speed $\\lambda$', fontsize=12)
    plt.title('Heatmap of tipping point $t^*(\\overline{C},\\lambda)$', fontsize=14)
    plt.tight_layout()
    plt.savefig('fig5_heatmap_tstar.png', dpi=300)
    plt.show()

    # ---------- Figure 6: Effect of topology (ensemble averaged agent-based) ----------
    topologies = ['complete', 'erdos_renyi', 'barabasi_albert']
    plt.figure(figsize=(10,6))
    for topo in topologies:
        print(f"Figure 6: averaging {p['ensemble_runs']} runs for topology {topo}...")
        G_topo = create_graph(topo, N=p['N'])
        costs_topo = compute_costs(G_topo, p['c_base'], p['c_frict'], p['gamma'])
        x_ensemble_topo = []
        t_ref_topo = None
        for run in range(p['ensemble_runs']):
            t_ag, x_ag = agent_based_simulation(G_topo, costs_topo,
                                                p['T_Q'], p['lambda_risk'], p['v'],
                                                p['alpha_bar'], p['beta'], p['kappa'],
                                                p['mu'], t_max=p['t_max'], dt=p['dt_agent'], x0=p['x0'])
            if t_ref_topo is None:
                t_ref_topo = t_ag
            x_ensemble_topo.append(x_ag)
        x_mean = np.mean(x_ensemble_topo, axis=0)
        plt.plot(t_ref_topo, x_mean, lw=2.5, label=f'{topo}')
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Fraction of PQC adopters $x(t)$', fontsize=12)
    plt.title('Effect of network topology on PQC diffusion (agent-based, ensemble average)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('fig6_topology_effect.png', dpi=300)
    plt.show()

    print("All figures generated successfully.")

if __name__ == "__main__":
    run_experiments()