import plotly.graph_objects as go
import os
import matplotlib.colors as mcolors

# 数据准备 (Data Preparation)
labels = [
    # 第一层：Sphere (大学科)
    "Hydrology", "Ecology and Biosciences", "Geology", "Solid Earth Geophysics", 
    "Geography", "Polar Science", "Atmospheric Sciences", "Oceanography",

    # 第二层：Hydrology 的 Sub-discipline
    "River & Estuarine Hydrology", "Groundwater Hydrology", "Limnology",

    # 第二层：Ecology and Biosciences 的 Sub-discipline
    "Aquatic & Limnological Ecology", "Landscape Ecology", "Ecosystem Ecology",

    # 第二层：Geology 的 Sub-discipline
    "Sedimentology", "Quaternary Geology", "Structural Geology",

    # 第二层：Solid Earth Geophysics 的 Sub-discipline
    "Seismology", "Tectonophysics", "Exploration Geophysics",

    # 第二层：Geography 的 Sub-discipline
    "Physical Geography", "Urban Geography", "Geomorphology",

    # 第二层：Polar Science 的 Sub-discipline
    "Glaciology", "Polar Climate Science", "Permafrost Science",

    # 第二层：Atmospheric Sciences 的 Sub-discipline
    "Meteorology", "Climatology", "Atmospheric Remote Sensing",

    # 第二层：Oceanography 的 Sub-discipline
    "Ocean Physics", "Ocean Geology", "Environmental Oceanography"
]

parents = [
    # 第一层：Sphere 的父级为空
    "", "", "", "", "", "", "", "",

    # 第二层：各 Sub-discipline 的父级是对应的 Sphere
    "Hydrology", "Hydrology", "Hydrology",
    "Ecology and Biosciences", "Ecology and Biosciences", "Ecology and Biosciences",
    "Geology", "Geology", "Geology",
    "Solid Earth Geophysics", "Solid Earth Geophysics", "Solid Earth Geophysics",
    "Geography", "Geography", "Geography",
    "Polar Science", "Polar Science", "Polar Science",
    "Atmospheric Sciences", "Atmospheric Sciences", "Atmospheric Sciences",
    "Oceanography", "Oceanography", "Oceanography"
]

values = [
    # 第一层：Sphere 的总计
    2034, 1140, 1581, 1262, 1691, 429, 1698, 965,

    # 第二层：各 Sub-discipline 的计数
    805, 790, 439,
    562, 298, 280,
    1068, 298, 215,
    845, 343, 74,
    1275, 276, 140,
    352, 46, 31,
    920, 619, 159,
    698, 163, 104
]

# 定义每个 Sphere 的主色
main_colors = {
    "Hydrology": "#9393FF",
    "Ecology and Biosciences": "#66B3FF",
    "Geology": "#C4C400",
    "Solid Earth Geophysics": "#FFAF60",
    "Geography": "#82D900",
    "Polar Science": "#81C0C0",
    "Atmospheric Sciences": "#C48888",
    "Oceanography": "#FF9D6F"
}

# 动态生成颜色
colors = []
for label in labels:
    if label in main_colors:  # 一级分类使用主色
        colors.append(main_colors[label])
    else:  # 二级分类生成渐变色
        parent = parents[labels.index(label)]
        base_color = mcolors.to_rgb(main_colors[parent])  # 获取父级的主色
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{parent}_gradient", [base_color, (1, 1, 1)], N=4
        )  # 从主色到白色生成渐变
        sub_index = sum(1 for p in parents[:labels.index(label)] if p == parent)  # 当前子分类的索引
        colors.append(mcolors.rgb2hex(cmap(sub_index / 3)))  # 根据索引分配颜色

# 创建旭日图 (Create Sunburst Chart)
fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",  # 确保子节点的值是手动提供的，不会被自动计算
    texttemplate="%{label}<br>%{percentRoot:.1%}",  # 显示相对于总值的百分比
    hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Percent of Total: %{percentRoot:.1%}<extra></extra>",
    insidetextorientation='radial',
    marker=dict(colors=colors)  # 动态生成的颜色
))

# 图表布局 (Chart Layout)
fig.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    width=1200,  # 增加宽度
    height=1200,  # 增加高度
    font=dict(
        size=30  # 设置全局字体大小
    )
)

# --- 保存图表为 PDF ---
output_dir = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/code/utils"  # 保存路径
output_filename = "sunburst_chart_updated.pdf"  # 保存为 PDF 文件
output_path = os.path.join(output_dir, output_filename)

print(f"尝试保存图表为 PDF 到: {output_path}")
try:
    # 保存为 PDF 文件
    fig.write_image(output_path)
    print("图表已成功保存为 PDF 文件。")
    print("提示：PDF 文件是矢量图形，放大时不会失真。")
except Exception as e:
    print(f"保存图表时出错: {e}")
    print("请确保目录存在、有写入权限，并且已安装 'kaleido' 包。")

# --- 保存图表为高分辨率图片 ---
output_filename_png = "sunburst_chart_high_res.png"  # 保存为高分辨率 PNG 文件
output_path_png = os.path.join(output_dir, output_filename_png)

print(f"尝试保存图表为高分辨率 PNG 到: {output_path_png}")
try:
    # 保存为高分辨率 PNG 文件
    fig.write_image(output_path_png, width=3000, height=3000)  # 设置宽度和高度为 3000 像素
    print("图表已成功保存为高分辨率 PNG 文件。")
    print("提示：PNG 文件是位图图形，适合高分辨率显示。")
except Exception as e:
    print(f"保存高分辨率图片时出错: {e}")
    print("请确保目录存在、有写入权限，并且已安装 'kaleido' 包。")
