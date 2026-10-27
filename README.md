# 丙氨酸二肽 well-tempered metadynamics

真空中 ACE–ALA–NME 的方法训练。引擎是 OpenMM，力场是 Amber14 ff14SB。这不是 G 蛋白偶联受体的计算，也没有做配体脱离。

φ 是唯一被偏置的变量。两条轨迹各 10 ns。按 `PROTOCOL.md` 里事先写定的标准，φ 自由能曲线通过了收敛检查。简历上对应的一句是“完成方法训练”。

说明在 `training_note.md`。自由能曲线、收敛图和数值结果在 `results/`。

`run_diala.py` 和 `draw_fes.py` 记录的是这次计算。输出路径写的是当时这台机器上的目录。图由 `draw_fes.py` 从 `results/fes_phi.csv` 和两条 hills 文件画出，因为这台机器上 matplotlib 的矩阵运算会中断。
