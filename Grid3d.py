import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# 创建均匀网格
grid = pv.ImageData(
    dimensions=(101, 101, 101),  # 网格点数（x, y, z）
    spacing=(1, 1, 1),           # 网格间距（米）
    origin=(0, 0, 0)             # 原点坐标
)
# 创建分层数据（基于单元中心坐标）
z_cells = np.indices((100, 100, 100))[2]  # 获取所有单元的z方向索引
z_center = z_cells + 0.5                 # 计算单元中心z坐标（米）

# 初始化层数据（三维数组）
layer = np.zeros_like(z_center, dtype=int)

# 设置分层条件
layer[z_center < 10] = 0    # 第一层：0-10米
layer[(z_center >= 10) & (z_center < 60)] = 1  # 第二层：10-60米
layer[z_center >= 60] = 2   # 第三层：60-100米

# 将数据添加到网格单元属性
grid.cell_data["layer"] = layer.ravel(order="F")  # 按Fortran顺序展平

# 可视化
plotter = pv.Plotter()
plotter.add_mesh(
    grid,
    scalars="layer",
    show_edges=False,
    opacity=0.7,
    cmap="jet",
    clim=[0, 2]
)
plotter.show_axes()
plotter.add_title("3D层状介质模型（单位：米）")
plotter.show()


