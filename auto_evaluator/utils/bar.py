import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = [
    "gemini-2.5-pro-thinking",
    "claude-3-7-sonnet-latest",
    "gpt-4o-2024-11-20",
    "gemini-2.5-flash",
    "qwen2.5-vl-72B",
    "internvl-2.5-78B"
]

# 数据：专业问题和感知问题的准确率
professional_problems = [56.31, 51.68, 50.45, 51.35, 41.43, 43.27]
perception_problems = [77.06, 78.11, 81.86, 75.86, 57.72, 58.02]

# 设置柱状图的宽度和位置
x = np.arange(len(models))  # 模型的索引
width = 0.35  # 柱子的宽度

# 创建柱状图
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")  # 设置背景为白色
bars1 = ax.bar(x - width/2, professional_problems, width, label="Professional Problems", color="#377483")  # 使用指定颜色
bars2 = ax.bar(x + width/2, perception_problems, width, label="Perception Problems", color="#C7DFF0")  # 使用指定颜色

# 添加标题和标签
ax.set_title("Model Accuracy: Professional vs Perception Problems", fontsize=16, pad=20)
ax.set_xlabel("Models", fontsize=12, labelpad=10)
ax.set_ylabel("Accuracy (%)", fontsize=12, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
ax.legend(fontsize=10, frameon=False)  # 去掉图例的边框

# 添加数值标签
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.2f}%", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.2f}%", ha="center", fontsize=9)

# 去掉顶部和右侧的边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 调整网格线样式
ax.grid(axis="y", linestyle="--", alpha=0.7)  # 仅显示水平网格线
ax.set_axisbelow(True)  # 确保网格线在柱状图下方

# 保存为 PDF 文件
plt.tight_layout()
plt.savefig("/fs-computility/ai4sData/zhaoxiangyu1/neurips_mmearth_benchmark/code/utils/Bar_Chart_Optimized.pdf", format="pdf", bbox_inches="tight")

# # 显示图形
# plt.show()
