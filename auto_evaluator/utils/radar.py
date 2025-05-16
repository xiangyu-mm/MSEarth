import matplotlib.pyplot as plt
import numpy as np

# Data (Keep as is)
labels_task1 = ['Hydrology', 'Geology', 'Geography', 'Polar Science', 'Atmospheric', 'Oceanography', 'Solid Earth Geophysics', 'Ecology and Biosciences']
data_task1 = [
    [51.21, 46.12, 51.38, 45.45, 43.14, 36.79, 48.31, 54.26],  # InternVL2.5-78B
    [48.38, 45.45, 46.46, 46.75, 45.12, 37.50, 43.96, 47.34],  # Qwen-2.5-VL-72B
    [55.67, 60.50, 59.08, 68.83, 55.21, 56.79, 57.49, 60.64],  # Claude-3.7-sonnet
    [56.68, 64.30, 59.08, 72.73, 50.41, 58.93, 56.52, 57.45],  # gpt-4o-2024-11-20
    [62.75, 62.31, 62.46, 63.64, 57.02, 60.00, 64.73, 62.77],  # gemini-2.5-pro-thinking
]
models_task1 = ['InternVL2.5-78B', 'Qwen-2.5-VL-72B', 'Claude-3.7-sonnet', 'gpt-4o-2024-11-20', 'gemini-2.5-pro-thinking']

labels_task2 = ['Hydrology', 'Geology', 'Geography', 'Polar Science', 'Atmospheric', 'Oceanography', 'Solid Earth Geophysics', 'Ecology and Biosciences']
data_task2 = [
    [42.69, 42.81, 45.56, 48.84, 47.19, 53.73, 47.83, 43.69],  # InternVL2.5-78B
    [41.54, 38.56, 43.33, 46.51, 43.82, 44.78, 46.09, 36.89],  # Qwen-2.5-VL-72B
    [48.46, 43.79, 47.22, 48.84, 49.44, 56.72, 51.30, 45.63],  # Claude-3.7-sonnet
    [45.77, 44.44, 48.89, 51.16, 52.06, 54.48, 52.17, 44.66],  # gpt-4o-2024-11-20
    [45.38, 44.44, 43.33, 48.84, 53.18, 52.24, 47.83, 49.51],  # gemini-2.5-pro-thinking
]
models_task2 = ['InternVL2.5-78B', 'Qwen-2.5-VL-72B', 'Claude-3.7-sonnet', 'gpt-4o-2024-11-20', 'gemini-2.5-pro-thinking']

# 设置雷达图绘制函数
def plot_radar(ax, labels, data, models, title, yticks, yticklabels, colors, line_styles, markers):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # Use default orientation: 0 degrees (first label 'Hydrology') is East (Right)

    # 绘制每个模型的雷达图
    for i, model_data in enumerate(data):
        values = model_data + model_data[:1]
        ax.plot(
            angles, values, label=models[i],
            linestyle=line_styles[i], color=colors[i], marker=markers[i], linewidth=2
        )
        ax.fill(angles, values, color=colors[i], alpha=0.05)

    # 设置标签和标题
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, color="grey", size=10) # Keep radial tick labels small
    ax.set_xticks(angles[:-1])
    # --- Increase font size for category labels ---
    ax.set_xticklabels(labels, fontsize=15) # Increased from 10 to 12
    # --- Adjust title padding if needed ---
    ax.set_title(title, size=18, pad=25) # Increased pad from 20 to 25
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# 定义样式
colors = ['blue', 'orange', 'green', 'red', 'purple']
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', 'D', '^', 'v']

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(polar=True), gridspec_kw={'wspace': 0.3})

# 绘制第一个任务的雷达图
plot_radar(
    axes[0],
    labels_task1,
    data_task1,
    models_task1,
    "MCQ Accuracy by Primary Sphere",
    yticks=[20, 40, 60, 80],
    yticklabels=['20%', '40%', '60%', '80%'],
    colors=colors,
    line_styles=line_styles,
    markers=markers
)

# 绘制第二个任务的雷达图
plot_radar(
    axes[1],
    labels_task2,
    data_task2,
    models_task2,
    "OE Accuracy by Primary Sphere",
    yticks=[10, 30, 50, 70],
    yticklabels=['10%', '30%', '50%', '70%'],
    colors=colors,
    line_styles=line_styles,
    markers=markers
)

# 获取 handles 和 labels 用于图例
handles, labels_legend = axes[0].get_legend_handles_labels()

# 创建 figure legend
fig.legend(
    handles,
    labels_legend,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.01), # Position below plots
    ncol=5,
    fontsize=10, # Keep legend font size as is unless specified otherwise
    frameon=False
)

# 保存为 PDF
save_path = "/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/code/utils/two_radar_charts.pdf"
try:
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Chart saved successfully to {save_path}")
except Exception as e:
    print(f"Error saving file: {e}")

# 显示图表 (取消注释以在屏幕上显示)
# plt.show()
