"""Draw the alanine-dipeptide free-energy figures.

matplotlib's renderer calls numpy.linalg, and those matrix kernels abort on
this machine. The curves themselves are the ones already written to fes_phi.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"D:\repositories\md_runs\diala_wtmetad")
ROOT = Path(__file__).resolve().parent
N_GRID = 361
GAMMA = 8.0
BLUE = (0, 114, 178)
ORANGE = (213, 94, 0)
INK = (20, 20, 20)
GRID = (220, 220, 220)


def font(size):
    return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", size)


def reconstruct(hills, sigma, step_limit):
    xs = np.linspace(-np.pi, np.pi, N_GRID)
    grid = np.zeros(N_GRID)
    for step, center, height in hills:
        if step > step_limit:
            continue
        delta = (xs - center + np.pi) % (2.0 * np.pi) - np.pi
        grid += height * np.exp(-0.5 * (delta / sigma) ** 2)
        grid[-1] = grid[0]
    free_energy = -(GAMMA / (GAMMA - 1.0)) * grid
    free_energy -= np.min(free_energy[:-1])
    return free_energy[:-1]


def load_curves():
    table = np.loadtxt(OUT / "fes_phi.csv", delimiter=",", skiprows=1)
    phi = table[:, 0]
    saved = {
        20261026: {5_000_000: table[:, 1], 2_500_000: table[:, 3]},
        20261027: {5_000_000: table[:, 2], 2_500_000: table[:, 4]},
    }
    series = {}
    for seed in (20261026, 20261027):
        data = np.load(OUT / f"hills_{seed}.npz")
        hills = np.asarray(data["hills"], dtype=float)
        sigma = float(data["sigma"])
        rebuilt = reconstruct(hills, sigma, 5_000_000)
        if float(np.max(np.abs(rebuilt - saved[seed][5_000_000]))) > 1e-6:
            raise SystemExit(f"rebuilt 10 ns curve does not match fes_phi.csv for {seed}")
        saved[seed][1_000_000] = reconstruct(hills, sigma, 1_000_000)
        series[seed] = np.asarray(data["phi"], dtype=float)
    return phi, saved, series


def minima(phi, free_energy):
    found = []
    for index in range(phi.size):
        left = free_energy[index - 1]
        right = free_energy[(index + 1) % phi.size]
        if free_energy[index] <= left and free_energy[index] < right:
            found.append((float(phi[index]), float(free_energy[index])))
    return sorted(found, key=lambda item: item[1])


def barrier(phi, free_energy, first, second):
    i = int(np.argmin(np.abs(phi - first)))
    j = int(np.argmin(np.abs(phi - second)))
    if i > j:
        i, j = j, i
    path_a = float(np.max(free_energy[i : j + 1]))
    path_b = float(max(np.max(free_energy[j:]), np.max(free_energy[: i + 1])))
    return min(path_a, path_b)


class Panel:
    def __init__(self, draw, box, xlim, ylim):
        self.draw = draw
        self.left, self.top, self.right, self.bottom = box
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim

    def x(self, value):
        return self.left + (value - self.xmin) / (self.xmax - self.xmin) * (self.right - self.left)

    def y(self, value):
        return self.bottom - (value - self.ymin) / (self.ymax - self.ymin) * (self.bottom - self.top)

    def frame(self, xticks, yticks, xlabel, ylabel, fonts):
        self.draw.line((self.left, self.top, self.left, self.bottom), fill=INK, width=2)
        self.draw.line((self.left, self.bottom, self.right, self.bottom), fill=INK, width=2)
        for tick in xticks:
            xx = self.x(tick)
            self.draw.line((xx, self.bottom, xx, self.bottom + 8), fill=INK, width=2)
            label = f"{tick:g}"
            self.draw.text((xx, self.bottom + 12), label, fill=INK, font=fonts[0], anchor="mt")
        for tick in yticks:
            yy = self.y(tick)
            self.draw.line((self.left - 8, yy, self.left, yy), fill=INK, width=2)
            self.draw.text((self.left - 12, yy), f"{tick:g}", fill=INK, font=fonts[0], anchor="rm")
        self.draw.text(
            ((self.left + self.right) / 2, self.bottom + 48),
            xlabel,
            fill=INK,
            font=fonts[1],
            anchor="mt",
        )
        self.draw.text((self.left, self.top - 14), ylabel, fill=INK, font=fonts[1], anchor="lb")

    def polyline(self, xs, ys, color, width=3, dash=None):
        points = [(self.x(float(x)), self.y(float(y))) for x, y in zip(xs, ys)]
        if dash is None:
            self.draw.line(points, fill=color, width=width, joint="curve")
            return
        for start in range(0, len(points) - 1, dash[0] + dash[1]):
            stop = min(start + dash[0], len(points))
            if stop - start >= 2:
                self.draw.line(points[start:stop], fill=color, width=width, joint="curve")


def save_svg(path, width, height, elements):
    body = "\n".join(elements)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n',
        encoding="utf-8",
    )


def svg_frame(box, xlim, ylim, xticks, yticks, xlabel, ylabel):
    left, top, right, bottom = box
    xmin, xmax = xlim
    ymin, ymax = ylim

    def x(value):
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def y(value):
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    parts = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#141414" stroke-width="2"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#141414" stroke-width="2"/>',
    ]
    for tick in xticks:
        xx = x(tick)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{bottom}" x2="{xx:.1f}" y2="{bottom + 8}" stroke="#141414" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{bottom + 28}" text-anchor="middle" font-family="Arial" font-size="22" fill="#141414">{tick:g}</text>'
        )
    for tick in yticks:
        yy = y(tick)
        parts.append(
            f'<line x1="{left - 8}" y1="{yy:.1f}" x2="{left}" y2="{yy:.1f}" stroke="#141414" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{left - 14}" y="{yy + 7:.1f}" text-anchor="end" font-family="Arial" font-size="22" fill="#141414">{tick:g}</text>'
        )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{bottom + 64}" text-anchor="middle" font-family="Arial" font-size="28" fill="#141414">{xlabel}</text>'
    )
    parts.append(
        f'<text x="{left}" y="{top - 16}" font-family="Arial" font-size="28" fill="#141414">{ylabel}</text>'
    )
    return parts, x, y


def svg_line(xs, ys, x, y, color, width=3, dash=None):
    points = " ".join(f"{x(float(a)):.2f},{y(float(b)):.2f}" for a, b in zip(xs, ys))
    style = f'fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'
    if dash:
        style += f' stroke-dasharray="{" ".join(str(item) for item in dash)}"'
    return f'<polyline points="{points}" {style}/>'


def write_note(summary):
    sentence = (
        "按事先写定的标准，两条 10 ns 曲线在低能区间一致，并且 φ 有来回跨越。"
        if summary["reads_as_converged"]
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
    phi, curves, series = load_curves()
    summary = json.loads((OUT / "convergence.json").read_text(encoding="utf-8"))
    mean = 0.5 * (curves[20261026][5_000_000] + curves[20261027][5_000_000])
    wells = minima(phi, mean)
    print("mean minima", [(round(a, 1), round(b, 2)) for a, b in wells[:4]])
    if len(wells) >= 2:
        print("barrier", round(barrier(phi, mean, wells[0][0], wells[1][0]), 2))
    ymax = float(max(np.max(curves[20261026][5_000_000]), np.max(curves[20261027][5_000_000])))
    ylim = (0.0, float(np.ceil((ymax + 1.0) / 10.0) * 10.0))
    yticks = list(range(0, int(ylim[1]) + 1, 10))
    xticks = [-180, -90, 0, 90, 180]
    fonts = (font(22), font(28), font(20))
    width, height = 1280, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panel = Panel(draw, (140, 90, 1180, 700), (-180, 180), ylim)
    panel.frame(xticks, yticks, "phi (degrees)", "free energy (kJ/mol)", fonts)
    panel.polyline(phi, curves[20261026][5_000_000], BLUE)
    panel.polyline(phi, curves[20261027][5_000_000], ORANGE)
    draw.line((150, 860, 230, 860), fill=BLUE, width=4)
    draw.text((240, 860), "seed 20261026, 10 ns", fill=INK, font=fonts[2], anchor="lm")
    draw.line((700, 860, 780, 860), fill=ORANGE, width=4)
    draw.text((790, 860), "seed 20261027, 10 ns", fill=INK, font=fonts[2], anchor="lm")
    image.save(OUT / "fes_phi.png", dpi=(200, 200))
    parts, sx, sy = svg_frame(
        (140, 90, 1180, 700), (-180, 180), ylim, xticks, yticks, "phi (degrees)", "free energy (kJ/mol)"
    )
    parts.append(svg_line(phi, curves[20261026][5_000_000], sx, sy, "#0072B2"))
    parts.append(svg_line(phi, curves[20261027][5_000_000], sx, sy, "#D55E00"))
    parts.append('<line x1="150" y1="860" x2="230" y2="860" stroke="#0072B2" stroke-width="4"/>')
    parts.append(
        '<text x="240" y="868" font-family="Arial" font-size="22" fill="#141414">seed 20261026, 10 ns</text>'
    )
    parts.append('<line x1="700" y1="860" x2="780" y2="860" stroke="#D55E00" stroke-width="4"/>')
    parts.append(
        '<text x="790" y="868" font-family="Arial" font-size="22" fill="#141414">seed 20261027, 10 ns</text>'
    )
    save_svg(OUT / "fes_phi.svg", width, height, parts)

    conv_w, conv_h = 1800, 860
    conv = Image.new("RGB", (conv_w, conv_h), "white")
    cdraw = ImageDraw.Draw(conv)
    left = Panel(cdraw, (150, 90, 860, 640), (-180, 180), ylim)
    left.frame(xticks, yticks, "phi (degrees)", "free energy (kJ/mol)", fonts)
    styles = {1_000_000: (8, 6), 2_500_000: (14, 6), 5_000_000: None}
    for seed, color in ((20261026, BLUE), (20261027, ORANGE)):
        for steps, dash in styles.items():
            left.polyline(phi, curves[seed][steps], color, dash=dash)
    time = np.arange(1, series[20261026].size + 1) * 0.0002
    right = Panel(cdraw, (1080, 90, 1700, 640), (0, 10), (-180, 180))
    right.frame([0, 2, 4, 6, 8, 10], [-180, -90, 0, 90, 180], "time (ns)", "phi (degrees)", fonts)
    for seed, color in ((20261026, BLUE), (20261027, ORANGE)):
        right.polyline(time, np.degrees(series[seed]), color, width=1)
    for level in (-0.5 * 180 / np.pi, 0.5 * 180 / np.pi):
        yy = right.y(level)
        cdraw.line((right.left, yy, right.right, yy), fill=(140, 140, 140), width=1)
    cdraw.text((150, 760), "solid 10 ns, long dash 5 ns, short dash 2 ns", fill=INK, font=fonts[2])
    cdraw.line((1080, 760, 1160, 760), fill=BLUE, width=4)
    cdraw.text((1170, 760), "20261026", fill=INK, font=fonts[2], anchor="lm")
    cdraw.line((1400, 760, 1480, 760), fill=ORANGE, width=4)
    cdraw.text((1490, 760), "20261027", fill=INK, font=fonts[2], anchor="lm")
    conv.save(OUT / "fes_convergence.png", dpi=(200, 200))

    parts, sx, sy = svg_frame(
        (150, 90, 860, 640), (-180, 180), ylim, xticks, yticks, "phi (degrees)", "free energy (kJ/mol)"
    )
    dash_of = {1_000_000: (8, 6), 2_500_000: (16, 8), 5_000_000: None}
    for seed, color in ((20261026, "#0072B2"), (20261027, "#D55E00")):
        for steps, dash in dash_of.items():
            parts.append(svg_line(phi, curves[seed][steps], sx, sy, color, dash=dash))
    parts2, sx2, sy2 = svg_frame(
        (1080, 90, 1700, 640), (0, 10), (-180, 180), [0, 2, 4, 6, 8, 10], [-180, -90, 0, 90, 180], "time (ns)", "phi (degrees)"
    )
    parts.extend(parts2)
    for seed, color in ((20261026, "#0072B2"), (20261027, "#D55E00")):
        parts.append(svg_line(time[::5], np.degrees(series[seed][::5]), sx2, sy2, color, width=1))
    for level in (-0.5 * 180 / np.pi, 0.5 * 180 / np.pi):
        yy = sy2(level)
        parts.append(f'<line x1="1080" y1="{yy:.1f}" x2="1700" y2="{yy:.1f}" stroke="#8c8c8c" stroke-width="1"/>')
    parts.append(
        '<text x="150" y="790" font-family="Arial" font-size="22" fill="#141414">solid 10 ns, long dash 5 ns, short dash 2 ns</text>'
    )
    save_svg(OUT / "fes_convergence.svg", conv_w, conv_h, parts)
    write_note(summary)
    print("wrote figures and training note")


if __name__ == "__main__":
    main()
