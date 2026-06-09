import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Paleta Wong
BLUE   = '#0072B2'
GREEN  = '#009E73'
ORANGE = '#E69F00'
PURPLE = '#CC79A7'

conditions = ['Baseline', '+SFT', '+CD', '+SFT + CD']
qwen    = [55.42, 38, 100, 71]
granite = [62.04, 48, 88,  75]

x = np.arange(len(conditions))
w = 0.32

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
})

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Barras
ax.bar(x - w/2, qwen,    w, label='Qwen 3.5 9B',       color=BLUE,  zorder=3)
ax.bar(x + w/2, granite, w, label='IBM Granite 4.1 8B', color=GREEN, zorder=3)

# Líneas de referencia con etiqueta directa (sin leyenda)
ax.axhline(100, color=ORANGE, linewidth=1.2, linestyle='--', zorder=2)
ax.axhline(100, color=PURPLE, linewidth=1.2, linestyle=':',  zorder=2)
ax.text(len(conditions) - 0.55, 101.5, 'GPT-5 / Gemini 3.1 Pro (baseline)',
        fontsize=9, color='#888888', va='bottom', ha='right')

# Anotaciones de significancia
sigs = [
    (0, qwen[0], granite[0], 'ns', 'ns'),
    (1, qwen[1], granite[1], 'ns', 'ns'),
    (2, qwen[2], granite[2], 'ns', 'ns'),
    (3, qwen[3], granite[3], 'ns', 'ns'),
]
for xi, vq, vg, sq, sg in sigs:
    if sq != 'ns':
        ax.text(xi - w/2, vq + 1.5, sq, ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=BLUE, zorder=4)
    if sg != 'ns':
        ax.text(xi + w/2, vg + 1.5, sg, ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=GREEN, zorder=4)

# Ejes
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylabel('Execution accuracy (%)', fontsize=11, labelpad=10)
ax.set_ylim(0, 113)
ax.set_yticks(range(0, 101, 20))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))

# Grid y spines
ax.grid(axis='y', color='#e0e0e0', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['bottom'].set_color('#cccccc')
ax.tick_params(axis='both', length=0, pad=6)

# Leyenda compacta arriba a la izquierda
legend = ax.legend(
    fontsize=10, frameon=True, loc='upper left',
    framealpha=0.9, edgecolor='#e0e0e0',
    handlelength=1.2, handleheight=0.8,
    borderpad=0.6, labelspacing=0.4,
)
legend.get_frame().set_linewidth(0.5)



plt.tight_layout()
plt.savefig('tritonbench_results.pdf', dpi=300, bbox_inches='tight')
plt.show()