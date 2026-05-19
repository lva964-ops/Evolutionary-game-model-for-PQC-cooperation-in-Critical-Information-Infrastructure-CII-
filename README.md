This repository contains the complete simulation code accompanying the research paper:
> **"Evolutionary game for PQC migration in CII: tipping points, systemic inertia, and network topology effects"**  
> Authors: [YV. Lakhno]  
The project models the diffusion of **Post-Quantum Cryptography (PQC)** among interdependent subjects of Critical Information Infrastructure (CII) in the face of the **Harvest Now, Decrypt Later (HNDL)** threat. Using **evolutionary game theory on graphs** and the **replicator equation**, we identify the analytical tipping point \( t^* \) where cooperation (PQC adoption) becomes evolutionarily stable, and demonstrate the existence of **systemic inertia** – a delay between the moment cooperation becomes advantageous and the time it dominates the population.

## Key features

- Weighted directed graphs with **incoming weight normalization** (sum of incoming dependencies = 1).
- Exponential HNDL risk \( R(t) = v e^{\lambda(t-T_Q)} \).
- Technological friction: migration cost depends on unweighted in‑degree (hubs pay more).
- Asynchronous strategy updating via the **Fermi rule**.
- Macroscopic **replicator equation** with a small mutation term \( \mu(1-2x) \).
- Agent‑based simulations with **spontaneous innovation** (mutation) to prevent early extinction.
- **Ensemble averaging** (20 runs) for robust, smooth curves.
- Generation of six publication‑ready figures (PNG, 300 dpi).
