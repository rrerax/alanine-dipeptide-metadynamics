"""Well-tempered metadynamics of alanine dipeptide in vacuum.

OpenMM implementation of the Lugano tutorial's scientific content.
This is method training. It is not a GPCR calculation and it does not use PLUMED.
Rules are in PROTOCOL.md and are not adjusted after the free-energy curves exist.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["OPENMM_CPU_THREADS"] = "2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openmm import (
    Continuous1DFunction,
    CustomCVForce,
    CustomTorsionForce,
    LangevinMiddleIntegrator,
    Platform,
    Vec3,
)
from openmm import unit
from openmm.app import (
    ForceField,
    NoCutoff,
    PDBFile,
    Simulation,
    Topology,
    element,
)
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent
OUT = Path(r"D:\repositories\md_runs\diala_wtmetad")
OUT.mkdir(parents=True, exist_ok=True)

TEMPERATURE = 300.0 * unit.kelvin
FRICTION = 1.0 / unit.picosecond
DT = 0.002 * unit.picoseconds
H0 = 1.2  # kJ/mol, peak height
PACE = 500
GAMMA = 8.0
KB = 0.008314462618  # kJ/mol/K
DELTA_T = (GAMMA - 1.0) * 300.0
N_GRID = 361
UNBIASED_STEPS = 500_000  # 1 ns
META_STEPS = 5_000_000  # 10 ns
SEEDS = (20261026, 20261027)
SIGMA_SEED = 20261026


def build_molecule():
    mol = Chem.MolFromSmiles("CC(=O)N[C@@H](C)C(=O)NC")
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if centers != [(4, "S")]:
        raise SystemExit(f"expected L-alanine, CIP was {centers}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SIGMA_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise SystemExit("RDKit could not embed alanine dipeptide")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return mol


def heavy_neighbors(atom):
    return [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]


def carbonyl_carbons(mol):
    found = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
                found.append(atom)
                break
    return found


def assign_names(mol):
    carbonyls = carbonyl_carbons(mol)
    if len(carbonyls) != 2:
        raise SystemExit(f"expected two carbonyls, found {len(carbonyls)}")
    ace_c = ala_c = None
    for atom in carbonyls:
        carbons = [nbr for nbr in heavy_neighbors(atom) if nbr.GetAtomicNum() == 6]
        if len(carbons) != 1:
            raise SystemExit("carbonyl carbon does not have one carbon neighbor")
        if len(heavy_neighbors(carbons[0])) == 1:
            ace_c = atom
        else:
            ala_c = atom
    if ace_c is None or ala_c is None:
        raise SystemExit("could not separate the acetyl and alanyl carbonyls")
    ace_ch3 = next(nbr for nbr in heavy_neighbors(ace_c) if nbr.GetAtomicNum() == 6)
    ace_o = next(nbr for nbr in heavy_neighbors(ace_c) if nbr.GetAtomicNum() == 8)
    ala_n = next(nbr for nbr in heavy_neighbors(ace_c) if nbr.GetAtomicNum() == 7)
    ala_ca = next(
        nbr
        for nbr in heavy_neighbors(ala_n)
        if nbr.GetAtomicNum() == 6 and nbr.GetIdx() != ace_c.GetIdx()
    )
    ala_cb = next(
        nbr
        for nbr in heavy_neighbors(ala_ca)
        if nbr.GetAtomicNum() == 6 and len(heavy_neighbors(nbr)) == 1
    )
    ala_o = next(nbr for nbr in heavy_neighbors(ala_c) if nbr.GetAtomicNum() == 8)
    nme_n = next(nbr for nbr in heavy_neighbors(ala_c) if nbr.GetAtomicNum() == 7)
    nme_ch3 = next(
        nbr
        for nbr in heavy_neighbors(nme_n)
        if nbr.GetAtomicNum() == 6 and nbr.GetIdx() != ala_c.GetIdx()
    )

    def hyd(atom):
        return sorted(
            (nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1),
            key=lambda item: item.GetIdx(),
        )

    names = {}
    names[ace_ch3.GetIdx()] = ("ACE", "CH3")
    names[ace_c.GetIdx()] = ("ACE", "C")
    names[ace_o.GetIdx()] = ("ACE", "O")
    for atom, label in zip(hyd(ace_ch3), ("HH31", "HH32", "HH33")):
        names[atom.GetIdx()] = ("ACE", label)
    names[ala_n.GetIdx()] = ("ALA", "N")
    names[hyd(ala_n)[0].GetIdx()] = ("ALA", "H")
    names[ala_ca.GetIdx()] = ("ALA", "CA")
    names[hyd(ala_ca)[0].GetIdx()] = ("ALA", "HA")
    names[ala_cb.GetIdx()] = ("ALA", "CB")
    for atom, label in zip(hyd(ala_cb), ("HB1", "HB2", "HB3")):
        names[atom.GetIdx()] = ("ALA", label)
    names[ala_c.GetIdx()] = ("ALA", "C")
    names[ala_o.GetIdx()] = ("ALA", "O")
    names[nme_n.GetIdx()] = ("NME", "N")
    names[hyd(nme_n)[0].GetIdx()] = ("NME", "H")
    names[nme_ch3.GetIdx()] = ("NME", "CH3")
    for atom, label in zip(hyd(nme_ch3), ("HH31", "HH32", "HH33")):
        names[atom.GetIdx()] = ("NME", label)
    if len(names) != mol.GetNumAtoms():
        raise SystemExit(f"named {len(names)} of {mol.GetNumAtoms()} atoms")
    return names


def build_topology(mol, names):
    order = ("ACE", "ALA", "NME")
    conf = mol.GetConformer()
    positions = [None] * mol.GetNumAtoms()
    for idx in range(mol.GetNumAtoms()):
        point = conf.GetAtomPosition(idx)
        positions[idx] = Vec3(point.x / 10.0, point.y / 10.0, point.z / 10.0)
    topology = Topology()
    chain = topology.addChain()
    residues = {name: topology.addResidue(name, chain) for name in order}
    rd_to_atom = {}
    ordered_positions = []
    for resname in order:
        for idx in range(mol.GetNumAtoms()):
            if names[idx][0] != resname:
                continue
            rd_atom = mol.GetAtomWithIdx(idx)
            rd_to_atom[idx] = topology.addAtom(
                names[idx][1],
                element.Element.getByAtomicNumber(rd_atom.GetAtomicNum()),
                residues[resname],
            )
            ordered_positions.append(positions[idx])
    for bond in mol.GetBonds():
        topology.addBond(
            rd_to_atom[bond.GetBeginAtomIdx()],
            rd_to_atom[bond.GetEndAtomIdx()],
        )
    index = {atom.name: atom.index for residue in topology.residues() for atom in residue.atoms()}
    # Disambiguate atoms that share a name by residue.
    by_res = {}
    for residue in topology.residues():
        for atom in residue.atoms():
            by_res[(residue.name, atom.name)] = atom.index
    phi = (
        by_res[("ACE", "C")],
        by_res[("ALA", "N")],
        by_res[("ALA", "CA")],
        by_res[("ALA", "C")],
    )
    psi = (
        by_res[("ALA", "N")],
        by_res[("ALA", "CA")],
        by_res[("ALA", "C")],
        by_res[("NME", "N")],
    )
    return topology, ordered_positions, phi, psi


def make_system(topology, phi_atoms, psi_atoms):
    forcefield = ForceField("amber14/protein.ff14SB.xml")
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=NoCutoff,
        constraints=app_hbonds(),
    )
    phi = CustomTorsionForce("theta")
    phi.addTorsion(*phi_atoms)
    psi = CustomTorsionForce("theta")
    psi.addTorsion(*psi_atoms)
    grid = np.zeros(N_GRID)
    table = Continuous1DFunction(grid.tolist(), -np.pi, np.pi, True)
    bias = CustomCVForce("u(phi)")
    bias.addCollectiveVariable("phi", phi)
    bias.addCollectiveVariable("psi", psi)
    bias.addTabulatedFunction("u", table)
    bias.setForceGroup(1)
    system.addForce(bias)
    return system, bias, table, grid


def app_hbonds():
    from openmm.app import HBonds

    return HBonds


def add_gaussian(grid, center, height, sigma):
    xs = np.linspace(-np.pi, np.pi, N_GRID)
    delta = (xs - center + np.pi) % (2.0 * np.pi) - np.pi
    grid += height * np.exp(-0.5 * (delta / sigma) ** 2)
    grid[-1] = grid[0]


class CVReporter:
    def __init__(self, bias, interval):
        self.bias = bias
        self.interval = interval
        self.phi = []
        self.psi = []

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return {"steps": steps, "periodic": False, "include": []}

    def report(self, simulation, state):
        phi, psi = self.bias.getCollectiveVariableValues(simulation.context)
        self.phi.append(float(phi))
        self.psi.append(float(psi))


def new_simulation(topology, system, choice, seed, positions):
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, DT)
    integrator.setRandomNumberSeed(seed)
    if choice["name"] == "CPU":
        platform = Platform.getPlatformByName("CPU")
        platform.setPropertyDefaultValue("Threads", "2")
        simulation = Simulation(topology, system, integrator, platform)
    else:
        platform = Platform.getPlatformByName("OpenCL")
        simulation = Simulation(
            topology,
            system,
            integrator,
            platform,
            {"OpenCLPlatformIndex": "0", "DeviceIndex": "0", "Precision": "mixed"},
        )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE, seed)
    return simulation


def choose_platform(topology, positions, phi_atoms, psi_atoms):
    timings = {}
    for name in ("CPU", "OpenCL"):
        system, bias, table, grid = make_system(topology, phi_atoms, psi_atoms)
        choice = {"name": name}
        simulation = new_simulation(topology, system, choice, SIGMA_SEED, positions)
        simulation.step(500)
        started = time.perf_counter()
        for _ in range(20):
            simulation.step(PACE)
        timings[name] = time.perf_counter() - started
        del simulation
    winner = min(timings, key=timings.get)
    record = {"seconds_for_20_calls": timings, "chosen": winner}
    (OUT / "platform.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("platform", record, flush=True)
    return {"name": winner}


def bias_energy(simulation):
    state = simulation.context.getState(getEnergy=True, groups={1})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def smoke_test(topology, positions, phi_atoms, psi_atoms, choice):
    system, bias, table, grid = make_system(topology, phi_atoms, psi_atoms)
    simulation = new_simulation(topology, system, choice, SIGMA_SEED, positions)
    simulation.minimizeEnergy(maxIterations=50)
    before = bias_energy(simulation)
    phi, _ = bias.getCollectiveVariableValues(simulation.context)
    add_gaussian(grid, float(phi), H0, 0.30)
    table.setFunctionParameters(grid.tolist(), -np.pi, np.pi)
    bias.updateParametersInContext(simulation.context)
    after = bias_energy(simulation)
    print(f"smoke bias energy {before:.4f} -> {after:.4f}", flush=True)
    if abs(before) > 1e-6 or not (1.0 <= after <= 1.4):
        raise SystemExit("tabulated bias did not match a single 1.2 kJ/mol hill")
    del simulation


def minimize(topology, positions, phi_atoms, psi_atoms, choice):
    system, bias, table, grid = make_system(topology, phi_atoms, psi_atoms)
    simulation = new_simulation(topology, system, choice, SIGMA_SEED, positions)
    simulation.minimizeEnergy(maxIterations=500)
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    minimized = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    print(f"minimized energy {energy:.2f} kJ/mol", flush=True)
    if not np.isfinite(energy):
        raise SystemExit("minimization energy is not finite")
    with open(OUT / "diala_minimized.pdb", "w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, state.getPositions(), handle)
    np.save(OUT / "minimized_positions_nm.npy", minimized)
    return minimized


def run_biased(topology, positions, phi_atoms, psi_atoms, choice, steps, seed, sigma, deposit):
    system, bias, table, grid = make_system(topology, phi_atoms, psi_atoms)
    simulation = new_simulation(topology, system, choice, seed, positions)
    reporter = CVReporter(bias, 100 if deposit else 50)
    simulation.reporters.append(reporter)
    hills = []
    done = 0
    started = time.perf_counter()
    while done < steps:
        chunk = min(PACE, steps - done)
        simulation.step(chunk)
        done += chunk
        if deposit and done % PACE == 0:
            phi, _ = bias.getCollectiveVariableValues(simulation.context)
            height = H0 * np.exp(-bias_energy(simulation) / (KB * DELTA_T))
            add_gaussian(grid, float(phi), height, sigma)
            table.setFunctionParameters(grid.tolist(), -np.pi, np.pi)
            bias.updateParametersInContext(simulation.context)
            hills.append((done, float(phi), float(height)))
        if done % 500_000 == 0:
            elapsed = time.perf_counter() - started
            print(f"seed {seed} step {done}/{steps} {elapsed:.0f}s", flush=True)
    return np.array(reporter.phi), np.array(reporter.psi), hills


def circular_std(phi):
    mean_sin = float(np.mean(np.sin(phi)))
    mean_cos = float(np.mean(np.cos(phi)))
    radius = min(1.0, float(np.hypot(mean_sin, mean_cos)))
    if radius <= 1e-12:
        return np.pi
    return float(np.sqrt(-2.0 * np.log(radius)))


def sigma_from_unbiased(phi):
    # Samples are every 0.1 ps. Drop the first 100 ps.
    keep = phi[1000:]
    raw = circular_std(keep)
    used = "all_after_100ps"
    if raw > 1.0:
        bins = np.linspace(-np.pi, np.pi,  int((2 * np.pi) / 0.5) + 1)
        counts, edges = np.histogram(keep, bins=bins)
        peak = int(np.argmax(counts))
        center = 0.5 * (edges[peak] + edges[peak + 1])
        delta = (keep - center + np.pi) % (2.0 * np.pi) - np.pi
        keep = keep[np.abs(delta) < 1.0]
        raw = circular_std(keep)
        used = "largest_bin"
    record = {"sigma_rad": raw, "samples": int(keep.size), "rule": used}
    if not 0.05 <= raw <= 0.8:
        (OUT / "sigma.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise SystemExit(f"sigma {raw:.4f} rad is outside the frozen window")
    (OUT / "sigma.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    np.save(OUT / "unbiased_phi.npy", phi)
    print("sigma", record, flush=True)
    return raw


def reconstruct(hills, sigma, step_limit=None):
    grid = np.zeros(N_GRID)
    for step, phi, height in hills:
        if step_limit is not None and step > step_limit:
            continue
        add_gaussian(grid, phi, height, sigma)
    free_energy = -(GAMMA / (GAMMA - 1.0)) * grid
    free_energy -= np.min(free_energy[:-1])
    return free_energy[:-1]


def transitions(phi, sample_ps):
    needed = max(1, int(round(1.0 / sample_ps)))
    current = None
    run = 0
    committed = None
    count = 0
    for value in phi:
        side = "neg" if value < -0.5 else "pos" if value > 0.5 else None
        if side is None or side != current:
            current = side
            run = 1 if side is not None else 0
        else:
            run += 1
        if side is not None and run >= needed:
            if committed is not None and side != committed:
                count += 1
            committed = side
    return count


def masked_rmsd(first, second, mask):
    if int(np.count_nonzero(mask)) < 5:
        return None
    a = first[mask] - np.mean(first[mask])
    b = second[mask] - np.mean(second[mask])
    return float(np.sqrt(np.mean((a - b) ** 2)))


def write_figures(phi_deg, curves, series, summary):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {20261026: "#0072B2", 20261027: "#D55E00"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for seed in SEEDS:
        ax.plot(phi_deg, curves[seed][META_STEPS], color=colors[seed], label=f"seed {seed}, 10 ns")
    ax.set_xlabel("φ (degrees)")
    ax.set_ylabel("free energy (kJ/mol)")
    ax.set_xlim(-180, 180)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fes_phi.png", dpi=200)
    fig.savefig(OUT / "fes_phi.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for seed in SEEDS:
        for steps, style in (
            (1_000_000, ":"),
            (2_500_000, "--"),
            (META_STEPS, "-"),
        ):
            axes[0].plot(
                phi_deg,
                curves[seed][steps],
                linestyle=style,
                color=colors[seed],
                label=f"{seed}, {steps * 0.002 / 1000:.0f} ns",
            )
    axes[0].set_xlabel("φ (degrees)")
    axes[0].set_ylabel("free energy (kJ/mol)")
    axes[0].set_xlim(-180, 180)
    axes[0].legend(frameon=False, fontsize=8)
    time_ns = np.arange(1, series[SEEDS[0]].size + 1) * 0.0002
    for seed in SEEDS:
        axes[1].plot(time_ns, np.degrees(series[seed]), color=colors[seed], lw=0.6, label=str(seed))
    axes[1].axhline(-0.5 * 180 / np.pi, color="0.5", lw=0.6)
    axes[1].axhline(0.5 * 180 / np.pi, color="0.5", lw=0.6)
    axes[1].set_xlabel("time (ns)")
    axes[1].set_ylabel("φ (degrees)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fes_convergence.png", dpi=200)
    fig.savefig(OUT / "fes_convergence.svg")
    plt.close(fig)


def write_note(summary):
    converged = summary["reads_as_converged"]
    sentence = (
        "按事先写定的标准，两条 10 ns 曲线在低能区间一致，并且 φ 有来回跨越。"
        if converged
        else "按事先写定的标准，这条 φ 曲线还不能当作收敛的自由能。"
    )
    text = f"""# 丙氨酸二肽，方法训练

完成方法训练。

这是真空中 ACE–ALA–NME 的 well-tempered metadynamics，用来练习集体变量、偏置和自由能还原。它不是 G 蛋白偶联受体的计算，也没有做配体脱离。机器上没有 GROMACS 和 PLUMED，所以引擎是 OpenMM {summary['openmm']}，力场是 Amber14 ff14SB，偏置写在 `CustomCVForce` 里。Lugano 教程用的是 GROMACS、PLUMED 和 AMBER99SB-ILDN。

φ 是唯一被偏置的变量。ψ 只记录。初始高斯高度 1.2 kJ/mol，每 1 ps 沉积一次，bias factor 是 8。无偏置 1 ns 给出的 σ 是 {summary['sigma_rad']:.3f} rad。两条轨迹各 10 ns，种子是 20261026 和 20261027。

{sentence} 种子 20261026 的 φ 跨越 {summary['transitions']['20261026']} 次，5 ns 和 10 ns 曲线的均方根差是 {summary['half_rmsd_kJ']['20261026']:.2f} kJ/mol。种子 20261027 的跨越是 {summary['transitions']['20261027']} 次，均方根差是 {summary['half_rmsd_kJ']['20261027']:.2f} kJ/mol。两条 10 ns 曲线的均方根差是 {summary['replica_rmsd_kJ']:.2f} kJ/mol。判定线是至少 2 次跨越，并且这些均方根差不超过 2 kJ/mol。

曲线的最低点被平移到了 0。纵轴是这条轨迹和这个力场下的 φ 自由能，不是实验值，也不能外推到受体配体。
"""
    (OUT / "training_note.md").write_text(text, encoding="utf-8")
    (ROOT / "training_note.md").write_text(text, encoding="utf-8")


def main():
    import openmm

    mol = build_molecule()
    names = assign_names(mol)
    topology, positions, phi_atoms, psi_atoms = build_topology(mol, names)
    print("atoms", topology.getNumAtoms(), "phi", phi_atoms, "psi", psi_atoms, flush=True)
    positions = minimize(topology, positions, phi_atoms, psi_atoms, {"name": "CPU"})
    choice = choose_platform(topology, positions, phi_atoms, psi_atoms)
    smoke_test(topology, positions, phi_atoms, psi_atoms, choice)
    unbiased, _, _ = run_biased(
        topology, positions, phi_atoms, psi_atoms, choice, UNBIASED_STEPS, SIGMA_SEED, 0.3, False
    )
    sigma = sigma_from_unbiased(unbiased)
    curves = {}
    series = {}
    transition_counts = {}
    half_rmsd = {}
    phi_deg = np.degrees(np.linspace(-np.pi, np.pi, N_GRID)[:-1])
    for seed in SEEDS:
        phi, psi, hills = run_biased(
            topology, positions, phi_atoms, psi_atoms, choice, META_STEPS, seed, sigma, True
        )
        np.savez(OUT / f"hills_{seed}.npz", hills=np.array(hills), phi=phi, psi=psi, sigma=sigma)
        series[seed] = phi
        curves[seed] = {}
        for steps in (1_000_000, 2_000_000, 2_500_000, 4_000_000, META_STEPS):
            curves[seed][steps] = reconstruct(hills, sigma, steps)
        final = curves[seed][META_STEPS]
        half = curves[seed][2_500_000]
        mask = final <= np.min(final) + 20.0
        half_rmsd[str(seed)] = masked_rmsd(half, final, mask)
        transition_counts[str(seed)] = transitions(phi, 0.2)
        print("seed", seed, "transitions", transition_counts[str(seed)], "half", half_rmsd[str(seed)], flush=True)
    mean_curve = 0.5 * (curves[SEEDS[0]][META_STEPS] + curves[SEEDS[1]][META_STEPS])
    replica_mask = mean_curve <= np.min(mean_curve) + 20.0
    replica_rmsd = masked_rmsd(curves[SEEDS[0]][META_STEPS], curves[SEEDS[1]][META_STEPS], replica_mask)
    readable = (
        replica_rmsd is not None
        and all(value is not None and value <= 2.0 for value in half_rmsd.values())
        and replica_rmsd <= 2.0
        and all(value >= 2 for value in transition_counts.values())
    )
    table = np.column_stack(
        [phi_deg]
        + [curves[seed][META_STEPS] for seed in SEEDS]
        + [curves[seed][2_500_000] for seed in SEEDS]
    )
    np.savetxt(
        OUT / "fes_phi.csv",
        table,
        delimiter=",",
        header="phi_deg,F_10ns_20261026,F_10ns_20261027,F_5ns_20261026,F_5ns_20261027",
        comments="",
    )
    summary = {
        "openmm": openmm.__version__,
        "sigma_rad": sigma,
        "gamma": GAMMA,
        "height_kJ": H0,
        "pace_steps": PACE,
        "transitions": transition_counts,
        "half_rmsd_kJ": half_rmsd,
        "replica_rmsd_kJ": replica_rmsd,
        "reads_as_converged": readable,
        "platform": choice["name"],
    }
    (OUT / "convergence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_figures(phi_deg, curves, series, summary)
    write_note(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
