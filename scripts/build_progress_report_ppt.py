import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "report_assets_20260320_defense"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def choose_template() -> Path:
    candidates = sorted(PROJECT_ROOT.glob("*.pptx"))
    if not candidates:
        raise FileNotFoundError("No .pptx template found in project root.")
    for p in candidates:
        if "20260309" in p.name:
            return p
    return candidates[0]


PPT_TEMPLATE = choose_template()
PPT_OUTPUT = PROJECT_ROOT / "史鹏程 20260320_模型建设进展汇报_答辩风格.pptx"

# Thesis-defense style palette
NAVY = "#0B1D39"
BLUE = "#1E4E8C"
CYAN = "#1B7F8E"
ORANGE = "#C97824"
GREEN = "#2D9D72"
RED = "#C74B55"
TEXT_DARK = "#10243E"
TEXT_MUTED = "#5A6E86"
BG_SOFT = "#F5F8FC"
BORDER = "#D6DFEA"


def setup_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.15
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_train_metrics() -> Dict:
    return read_json(PROJECT_ROOT / "outputs" / "run_20260320_085725" / "train_metrics.json")


def load_fewshot_summary() -> List[Dict[str, str]]:
    return read_csv_rows(PROJECT_ROOT / "outputs" / "fewshot_bundle_cea_v1" / "bundle_summary.csv")


def load_real_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    data_dir = PROJECT_ROOT / "data" / "real_fewshot_cea"
    spectra = np.load(data_dir / "spectra.npy").astype(np.float32)
    labels = np.load(data_dir / "labels.npy").astype(np.int64)
    wavelengths = np.load(data_dir / "wavelengths.npy").astype(np.float32)
    label_map = read_json(data_dir / "label_map.json")["id_to_label"]
    id_to_label = {int(k): str(v) for k, v in label_map.items()}
    return spectra, labels, wavelengths, id_to_label


def build_pipeline_asset(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 4.8), dpi=240)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 0), 100, 34, facecolor="#F8FAFD", edgecolor="none"))

    stages = [
        (3, 8, 20, 17, BLUE, "阶段 A  理论预训练", "Mie 光谱生成\n物理噪声增强\n1D-ResNet Encoder", "已实现"),
        (28, 8, 20, 17, CYAN, "阶段 B  真实域适配", "成对光谱导入\nfew-shot 适配\n多 episode 评估", "已实现"),
        (53, 8, 20, 17, ORANGE, "阶段 C  物理约束判定", "红移方向检查\n饱和边界约束\n传统峰值交叉验证", "待实现"),
        (78, 8, 19, 17, NAVY, "阶段 D  工程化部署", "软件入口集成\n在线推理与日志\n自动报告闭环", "待实现"),
    ]

    for x, y, w, h, color, title, body, tag in stages:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + 0.5, y - 0.5),
                w,
                h,
                boxstyle="round,pad=0.7,rounding_size=2.0",
                linewidth=0,
                edgecolor="none",
                facecolor="#E9EEF6",
                alpha=0.5,
            )
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.7,rounding_size=2.0",
                linewidth=1.8,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.text(x + 1.3, y + h - 2.8, title, fontsize=14.8, color=color, fontweight="bold", va="top")
        ax.text(x + 1.3, y + 3.1, body, fontsize=11.6, color=TEXT_DARK, va="bottom", linespacing=1.5)
        badge_color = GREEN if tag == "已实现" else ORANGE
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + w - 7.0, y + h - 4.3),
                5.8,
                2.1,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                linewidth=0,
                facecolor=badge_color,
            )
        )
        ax.text(x + w - 4.1, y + h - 3.25, tag, fontsize=10.4, color="white", ha="center", va="center", fontweight="bold")

    for x in [23, 48, 73]:
        ax.annotate("", xy=(x + 3.7, 16.5), xytext=(x, 16.5), arrowprops=dict(arrowstyle="->", lw=2.2, color=TEXT_MUTED))

    ax.text(2.2, 30.0, "模型闭环：理论预训练 -> 真实域 few-shot -> 物理约束 -> 软件部署", fontsize=17, color=TEXT_DARK, fontweight="bold")
    ax.text(2.2, 26.9, "当前状态：A/B 已落地并完成实测；C/D 为下一阶段核心工作。", fontsize=12.2, color=TEXT_MUTED)
    fig.savefig(path)
    plt.close(fig)


def build_pretrain_asset(path: Path, train_metrics: Dict) -> None:
    history = train_metrics["history"]
    epochs = np.asarray([int(x["epoch"]) for x in history], dtype=np.int64)
    train_loss = np.asarray([float(x["train_loss"]) for x in history], dtype=np.float64)
    val_loss = np.asarray([float(x["val_loss"]) for x in history], dtype=np.float64)
    val_acc = np.asarray([float(x["val_acc"]) for x in history], dtype=np.float64) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=240)

    axes[0].plot(epochs, train_loss, marker="o", lw=2.4, color=BLUE, label="train loss")
    axes[0].plot(epochs, val_loss, marker="s", lw=2.4, color=ORANGE, label="val loss")
    axes[0].fill_between(epochs, 0, train_loss, color=BLUE, alpha=0.08)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("预训练损失曲线")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3, linestyle="--")

    axes[1].plot(epochs, val_acc, marker="o", lw=2.6, color=CYAN)
    best_idx = int(np.argmax(val_acc))
    axes[1].scatter([epochs[best_idx]], [val_acc[best_idx]], color=RED, s=60, zorder=5)
    axes[1].annotate(
        f"Best {val_acc[best_idx]:.2f}%",
        xy=(epochs[best_idx], val_acc[best_idx]),
        xytext=(epochs[best_idx] - 2.2, val_acc[best_idx] - 13),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
        fontsize=10.5,
        color=RED,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("预训练验证精度")
    axes[1].set_ylim(0, 102)
    axes[1].grid(alpha=0.3, linestyle="--")

    best = train_metrics["best"]
    fig.suptitle(
        f"Stage A 预训练结果：val_acc={best['val_acc']*100:.2f}%  val_loss={best['val_loss']:.3f}",
        fontsize=15.5,
        color=TEXT_DARK,
        fontweight="bold",
    )
    fig.savefig(path)
    plt.close(fig)


def build_real_data_asset(path: Path, spectra: np.ndarray, labels: np.ndarray, wavelengths: np.ndarray, id_to_label: Dict[int, str]) -> None:
    uniq, cnt = np.unique(labels, return_counts=True)
    order = np.argsort(cnt)[::-1]
    classes = [id_to_label[int(uniq[i])] for i in order]
    counts = [int(cnt[i]) for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=240)

    colors = [BLUE if c >= 6 else ORANGE for c in counts]
    bars = axes[0].bar(np.arange(len(classes)), counts, color=colors, alpha=0.92, edgecolor="white", linewidth=0.6)
    axes[0].axhline(4, color=CYAN, linestyle="--", lw=1.8, label="k=3,n_query=1 下限")
    axes[0].axhline(6, color=ORANGE, linestyle="--", lw=1.8, label="k=5,n_query=1 下限")
    axes[0].set_xticks(np.arange(len(classes)))
    axes[0].set_xticklabels(classes, rotation=40, ha="right")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("真实数据类别分布")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.35, str(val), ha="center", va="bottom", fontsize=9.2, color=TEXT_DARK)

    focus_ids = [0, 2, 4, 7, 8, 10]
    cmap = plt.colormaps["viridis"].resampled(len(focus_ids))
    for idx, cid in enumerate(focus_ids):
        sample_idx = np.where(labels == cid)[0]
        if sample_idx.size == 0:
            continue
        mean_curve = spectra[sample_idx].mean(axis=0)
        std_curve = spectra[sample_idx].std(axis=0)
        color = cmap(idx)
        axes[1].plot(wavelengths, mean_curve, lw=2.0, color=color, label=id_to_label[cid])
        axes[1].fill_between(wavelengths, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.14)

    axes[1].set_title("不同浓度的平均光谱与波动范围")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Normalized delta signal")
    axes[1].legend(frameon=False, fontsize=8.8, ncol=2)
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("Stage B 真实数据概览：168 样本 / 11 类 / 400 采样点（类别不均衡）", fontsize=15.5, color=TEXT_DARK, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def build_fewshot_asset(path: Path, rows: List[Dict[str, str]]) -> None:
    valid = [r for r in rows if r.get("status") == "ok"]
    grouped = {"prototype": [], "linear_head": []}
    for r in valid:
        grouped[r["mode"]].append(r)
    for mode in grouped:
        grouped[mode].sort(key=lambda x: int(x["k_shot"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=240)
    mode_color = {"prototype": BLUE, "linear_head": ORANGE}

    for mode, vals in grouped.items():
        if not vals:
            continue
        ks = np.asarray([int(v["k_shot"]) for v in vals], dtype=np.int64)
        acc = np.asarray([float(v["acc_mean"]) * 100 for v in vals], dtype=np.float64)
        acc_std = np.asarray([float(v["acc_std"]) * 100 for v in vals], dtype=np.float64)
        f1 = np.asarray([float(v["macro_f1_mean"]) * 100 for v in vals], dtype=np.float64)
        f1_std = np.asarray([float(v["macro_f1_std"]) * 100 for v in vals], dtype=np.float64)

        axes[0].errorbar(ks, acc, yerr=acc_std, marker="o", capsize=4, lw=2.3, color=mode_color[mode], label=mode)
        axes[1].errorbar(ks, f1, yerr=f1_std, marker="o", capsize=4, lw=2.3, color=mode_color[mode], label=mode)

    random_baseline = {1: 100 / 11, 3: 100 / 9, 5: 100 / 8}
    axes[0].plot(list(random_baseline.keys()), list(random_baseline.values()), linestyle="--", color=TEXT_MUTED, lw=1.8, label="random baseline")

    axes[0].set_title("Few-shot Query Accuracy")
    axes[1].set_title("Few-shot Macro-F1")
    for ax in axes:
        ax.set_xlabel("K-shot")
        ax.set_ylabel("Score (%)")
        ax.set_xticks([1, 3, 5])
        ax.set_ylim(0, 40)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(frameon=False, fontsize=9.5)

    fig.suptitle("Stage B few-shot 结果：已高于随机基线，但整体表现仍偏低", fontsize=15.5, color=TEXT_DARK, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def generate_assets() -> Dict[str, Path]:
    setup_matplotlib()
    train_metrics = load_train_metrics()
    fewshot_rows = load_fewshot_summary()
    spectra, labels, wavelengths, id_to_label = load_real_dataset()

    assets = {
        "pipeline": OUTPUT_ROOT / "pipeline_overview.png",
        "pretrain": OUTPUT_ROOT / "pretrain_summary.png",
        "real_data": OUTPUT_ROOT / "real_data_overview.png",
        "fewshot": OUTPUT_ROOT / "fewshot_summary.png",
        "tsne": PROJECT_ROOT / "outputs" / "run_20260320_085725" / "tsne_validation.png",
    }

    build_pipeline_asset(assets["pipeline"])
    build_pretrain_asset(assets["pretrain"], train_metrics)
    build_real_data_asset(assets["real_data"], spectra, labels, wavelengths, id_to_label)
    build_fewshot_asset(assets["fewshot"], fewshot_rows)
    return assets


def rgb(hex_code: str) -> RGBColor:
    raw = hex_code.replace("#", "")
    return RGBColor(int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def remove_all_slides(prs: Presentation) -> None:
    for idx in range(len(prs.slides) - 1, -1, -1):
        slide_id = prs.slides._sldIdLst[idx]
        prs.part.drop_rel(slide_id.rId)
        del prs.slides._sldIdLst[idx]


def set_title(slide, text: str) -> None:
    shape = slide.shapes.title
    shape.text = text
    p = shape.text_frame.paragraphs[0]
    p.font.name = "Microsoft YaHei"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = rgb(TEXT_DARK)


def add_text_card(
    slide,
    left,
    top,
    width,
    height,
    text,
    font_size=14,
    color=TEXT_DARK,
    bold=False,
    align=PP_ALIGN.LEFT,
    fill=None,
    line=BORDER,
    radius=True,
):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        left,
        top,
        width,
        height,
    )
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(1.0)
    else:
        shape.line.color.rgb = rgb("FFFFFF")
        shape.line.transparency = 1.0

    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Pt(10)
    tf.margin_right = Pt(10)
    tf.margin_top = Pt(8)
    tf.margin_bottom = Pt(8)
    tf.vertical_anchor = MSO_ANCHOR.TOP

    first = True
    for line_text in text.split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.text = line_text
        p.alignment = align
        p.font.name = "Microsoft YaHei"
        p.font.size = Pt(font_size)
        p.font.bold = bold if first else False
        p.font.color.rgb = rgb(color)
        first = False
    return shape


def add_header_tag(slide, text: str) -> None:
    add_text_card(slide, Cm(24.3), Cm(0.45), Cm(6.0), Cm(0.95), text, font_size=9.5, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER, fill=BLUE, line=BLUE)


def add_footer(slide, page_no: int) -> None:
    add_text_card(slide, Cm(0.8), Cm(18.1), Cm(22.0), Cm(0.8), "LSPR 模型建设进展汇报", font_size=9.5, color=TEXT_MUTED, line=None, radius=False)
    add_text_card(slide, Cm(31.0), Cm(18.0), Cm(1.0), Cm(0.9), str(page_no), font_size=10, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER, fill=BLUE, line=BLUE)


def add_picture(slide, path: Path, left, top, width, height) -> None:
    slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    frame = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    frame.fill.background()
    frame.line.color.rgb = rgb(BORDER)
    frame.line.width = Pt(1.1)


def add_bullets_card(slide, left, top, width, height, title: str, bullets: List[str], fill=BG_SOFT):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(BORDER)
    shape.line.width = Pt(1.0)

    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Pt(10)
    tf.margin_right = Pt(10)
    tf.margin_top = Pt(8)
    tf.margin_bottom = Pt(8)

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.name = "Microsoft YaHei"
    p0.font.size = Pt(15.5)
    p0.font.bold = True
    p0.font.color.rgb = rgb(TEXT_DARK)

    for line in bullets:
        p = tf.add_paragraph()
        p.text = line
        p.level = 0
        p.bullet = True
        p.font.name = "Microsoft YaHei"
        p.font.size = Pt(12.2)
        p.font.color.rgb = rgb(TEXT_MUTED)


def set_cell_text(cell, text: str, size: float = 11.0, bold: bool = False, color: str = TEXT_DARK, align: PP_ALIGN = PP_ALIGN.LEFT) -> None:
    cell.text = ""
    tf = cell.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Pt(6)
    tf.margin_right = Pt(6)
    tf.margin_top = Pt(3)
    tf.margin_bottom = Pt(3)
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = "Microsoft YaHei"
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = rgb(color)


def add_progress_table(slide, left, top, width, height) -> None:
    rows = [
        ("A1", "理论光谱生成", "已实现", "Mie 光谱生成与光学常数读取已完成"),
        ("A2", "物理噪声增强", "已实现", "高斯噪声、基线漂移、峰位/FWHM 扰动已完成"),
        ("A3", "Encoder 预训练", "已实现", "1D-ResNet 预训练完成，最佳 val_acc=98.91%"),
        ("A4", "表征验证", "已实现", "t-SNE 与关键指标已输出"),
        ("B1", "真实数据导入", "已实现", "CSV 与成对 Excel 数据已可直接处理"),
        ("B2", "few-shot 适配", "已实现", "prototype 与 linear_head 均已跑通"),
        ("B3", "多 episode 评估", "已实现", "K=1/3/5 汇总实验已完成"),
        ("B4", "单样本推理", "已实现", "离线单样本推理流程已完成"),
        ("C1", "物理约束判定", "待实现", "红移方向与边界约束尚未接入"),
        ("C2", "可信度输出", "待实现", "校准与不确定性解释尚未形成"),
        ("D1", "软件集成", "待实现", "尚未接入目标测试软件入口"),
        ("D2", "自动报告闭环", "待实现", "日志数据库与报告流程未完成"),
    ]

    headers = ["编码", "模块", "状态", "说明"]
    table_shape = slide.shapes.add_table(len(rows) + 1, 4, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Cm(2.0)
    table.columns[1].width = Cm(5.0)
    table.columns[2].width = Cm(3.0)
    table.columns[3].width = Cm(19.8)

    header_fill = rgb(NAVY)
    for c, head in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        set_cell_text(cell, head, size=12, bold=True, color="#FFFFFF", align=PP_ALIGN.CENTER)

    for r, row in enumerate(rows, start=1):
        bg = "#FBFDFF" if r % 2 == 1 else "#FFFFFF"
        for c in range(4):
            table.cell(r, c).fill.solid()
            table.cell(r, c).fill.fore_color.rgb = rgb(bg)

        set_cell_text(table.cell(r, 0), row[0], size=11.2, bold=True, align=PP_ALIGN.CENTER)
        set_cell_text(table.cell(r, 1), row[1], size=11.2)

        status = row[2]
        status_cell = table.cell(r, 2)
        if status == "已实现":
            status_cell.fill.solid()
            status_cell.fill.fore_color.rgb = rgb(GREEN)
            set_cell_text(status_cell, status, size=10.8, bold=True, color="#FFFFFF", align=PP_ALIGN.CENTER)
        else:
            status_cell.fill.solid()
            status_cell.fill.fore_color.rgb = rgb(ORANGE)
            set_cell_text(status_cell, status, size=10.8, bold=True, color="#FFFFFF", align=PP_ALIGN.CENTER)

        set_cell_text(table.cell(r, 3), row[3], size=11.0, color=TEXT_MUTED)


def add_content_slide(prs: Presentation, title: str, page_no: int):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    set_title(slide, title)
    add_header_tag(slide, "LSPR 答辩汇报")
    add_footer(slide, page_no)
    return slide


def add_title_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    add_text_card(slide, Cm(1.0), Cm(2.7), Cm(17.0), Cm(1.8), "汇报人：史鹏程", font_size=18, color=TEXT_MUTED, bold=True, line=None, radius=False)
    add_text_card(
        slide,
        Cm(1.0),
        Cm(3.8),
        Cm(27.5),
        Cm(3.8),
        "基于物理先验与 Few-shot 迁移的 LSPR 光谱智能分析\n模型建设进展汇报",
        font_size=24,
        color=TEXT_DARK,
        bold=True,
        line=None,
        radius=False,
    )
    add_text_card(slide, Cm(1.0), Cm(8.5), Cm(12.0), Cm(1.6), "汇报日期：2026.03.20", font_size=13.5, color=TEXT_MUTED, line=None, radius=False)
    add_text_card(
        slide,
        Cm(1.0),
        Cm(14.3),
        Cm(28.5),
        Cm(2.6),
        "汇报目标：系统梳理模型“计划实现”与“已经实现”的边界，明确现阶段学术结论与下一阶段关键工作。",
        font_size=14.3,
        color=TEXT_DARK,
        fill="#ECF3FB",
        line=BORDER,
    )
    add_footer(slide, 1)


def build_presentation(assets: Dict[str, Path]) -> None:
    prs = Presentation(str(PPT_TEMPLATE))
    remove_all_slides(prs)

    add_title_slide(prs)

    slide = add_content_slide(prs, "研究目标与科学问题", 2)
    add_bullets_card(
        slide,
        Cm(0.8),
        Cm(3.1),
        Cm(10.3),
        Cm(7.0),
        "核心科学问题",
        [
            "低浓度真实信号弱，容易被噪声与芯片差异掩盖。",
            "峰位、峰强、半高宽等特征强耦合，单变量难以定量。",
            "真实标注样本规模小，难以直接训练深层网络。",
        ],
    )
    add_bullets_card(
        slide,
        Cm(11.4),
        Cm(3.1),
        Cm(10.3),
        Cm(7.0),
        "建模策略",
        [
            "先用理论光谱和物理扰动训练可迁移 Encoder。",
            "再用真实数据做 few-shot 适配与泛化评估。",
            "最后引入物理规则与可信度约束并完成工程化。",
        ],
        fill="#F4FAF9",
    )
    add_text_card(
        slide,
        Cm(22.1),
        Cm(3.1),
        Cm(9.9),
        Cm(7.0),
        "评价口径\n\nStage A：表征可分性与训练稳定性\nStage B：真实 few-shot query 表现\nStage C/D：物理一致性与可交付性",
        font_size=13.6,
        color=TEXT_DARK,
        fill="#FFF7EE",
        line=BORDER,
    )
    add_text_card(
        slide,
        Cm(0.8),
        Cm(11.0),
        Cm(31.2),
        Cm(4.2),
        "本汇报重点是“模型能力边界”而不是“单一指标高低”：要明确哪些工作已经实证成立，哪些仍处于待验证阶段。",
        font_size=14.2,
        color=TEXT_DARK,
        fill="#ECF3FB",
        line=BORDER,
    )

    slide = add_content_slide(prs, "模型闭环与技术路线", 3)
    add_picture(slide, assets["pipeline"], Cm(0.9), Cm(3.0), Cm(30.4), Cm(11.0))
    add_text_card(
        slide,
        Cm(1.0),
        Cm(14.6),
        Cm(29.8),
        Cm(2.1),
        "闭环逻辑：先构建可解释表征，再完成真实域迁移，随后加入物理约束，最终形成可部署的软件推理链路。",
        font_size=13.6,
        color=TEXT_DARK,
        fill="#F7FAFD",
        line=BORDER,
    )

    slide = add_content_slide(prs, "实现进度总览（表格）", 4)
    add_progress_table(slide, Cm(0.8), Cm(3.0), Cm(30.8), Cm(12.4))
    add_text_card(
        slide,
        Cm(0.8),
        Cm(15.7),
        Cm(30.8),
        Cm(1.5),
        "结论：当前已完成从理论预训练到真实 few-shot 评估的主干流程；物理约束层与工程交付层尚未完成。",
        font_size=12.8,
        color=TEXT_DARK,
        fill="#ECF3FB",
        line=BORDER,
    )

    slide = add_content_slide(prs, "阶段 A：理论预训练已完成", 5)
    add_bullets_card(
        slide,
        Cm(0.8),
        Cm(3.0),
        Cm(10.0),
        Cm(5.0),
        "已实现内容",
        [
            "理论光谱生成与物理噪声增强流程已完成。",
            "1D-ResNet 预训练流程稳定收敛。",
            "可复用 Encoder 与可视化输出已形成。",
        ],
    )
    add_text_card(
        slide,
        Cm(0.8),
        Cm(8.4),
        Cm(10.0),
        Cm(3.8),
        "关键结果\n\n最佳验证精度：98.91%\n最佳验证损失：0.062\n表征维度：128",
        font_size=13.8,
        color=TEXT_DARK,
        fill="#F4FAF9",
        line=BORDER,
    )
    add_picture(slide, assets["pretrain"], Cm(11.1), Cm(3.0), Cm(10.2), Cm(5.5))
    add_picture(slide, assets["tsne"], Cm(21.6), Cm(3.0), Cm(9.2), Cm(9.1))
    add_text_card(
        slide,
        Cm(11.1),
        Cm(8.9),
        Cm(10.2),
        Cm(3.3),
        "解释：仿真域表现很强，说明表征学习成功；但真实域 few-shot 仍需独立评估，不能直接外推。",
        font_size=12.5,
        color=TEXT_MUTED,
        fill="#FFF7EE",
        line=BORDER,
    )

    slide = add_content_slide(prs, "阶段 B：真实数据接入与适配", 6)
    add_bullets_card(
        slide,
        Cm(0.8),
        Cm(3.0),
        Cm(10.2),
        Cm(6.6),
        "已完成能力",
        [
            "真实光谱导入流程已支持 CSV 与成对光谱格式。",
            "few-shot 适配支持 prototype 与 linear-head 两种模式。",
            "多 K-shot、多 episode 评估流程已自动化。",
            "单样本离线推理流程已可执行。",
        ],
    )
    add_text_card(
        slide,
        Cm(0.8),
        Cm(10.0),
        Cm(10.2),
        Cm(4.0),
        "当前数据规模\n\n总样本数：168\n类别数：11\n每条光谱采样点：400\n类别分布明显不均衡",
        font_size=13.4,
        color=TEXT_DARK,
        fill="#F4FAF9",
        line=BORDER,
    )
    add_picture(slide, assets["real_data"], Cm(11.4), Cm(3.0), Cm(19.5), Cm(11.0))

    slide = add_content_slide(prs, "阶段 B：few-shot 实验结果", 7)
    add_picture(slide, assets["fewshot"], Cm(0.9), Cm(3.0), Cm(18.8), Cm(10.9))
    add_bullets_card(
        slide,
        Cm(20.0),
        Cm(3.0),
        Cm(11.0),
        Cm(5.6),
        "结果摘要",
        [
            "最佳 accuracy：prototype + k=1，约 26.7%。",
            "最佳 macro-F1：prototype + k=1，约 20.4%。",
            "linear-head 未稳定超过 prototype，说明支持集规模仍受限。",
            "整体显著高于随机基线，但离部署可用仍有差距。",
        ],
        fill="#F7FAFD",
    )
    add_text_card(
        slide,
        Cm(20.0),
        Cm(8.9),
        Cm(11.0),
        Cm(5.0),
        "口径说明\n\n图中准确率是 few-shot episode 的 query 集准确率，不是全量测试集精度。\nprototype 更接近 Encoder 表征质量；linear-head 表示 Encoder + 适配头的综合效果。",
        font_size=12.8,
        color=TEXT_DARK,
        fill="#FFF7EE",
        line=BORDER,
    )

    slide = add_content_slide(prs, "当前瓶颈与待完成模块", 8)
    add_text_card(
        slide,
        Cm(0.9),
        Cm(3.1),
        Cm(10.0),
        Cm(5.8),
        "当前瓶颈\n\n1. 仿真域与真实域仍有明显 domain gap。\n2. 类别不均衡导致高 K-shot 可用类别减少。\n3. 细粒度 11 类定义增加学习难度。\n4. 支持集过小导致适配头学习不稳定。",
        font_size=13.0,
        color=TEXT_DARK,
        fill="#F7FAFD",
        line=BORDER,
    )
    add_text_card(
        slide,
        Cm(11.2),
        Cm(3.1),
        Cm(10.0),
        Cm(5.8),
        "待完成模块\n\n1. 物理约束判定层（红移方向、饱和边界）。\n2. 可信度与校准输出（可解释不确定性）。\n3. 软件入口对接与稳定推理封装。\n4. 自动日志与报告闭环。",
        font_size=13.0,
        color=TEXT_DARK,
        fill="#FFF7EE",
        line=BORDER,
    )
    add_text_card(
        slide,
        Cm(21.5),
        Cm(3.1),
        Cm(9.6),
        Cm(5.8),
        "学术结论\n\n当前可以确认“已形成可复现实验基线”，但不能宣称“已达到应用级别”。\n下一阶段关键是提升真实任务定义与物理一致性，而非单纯增加脚本数量。",
        font_size=13.0,
        color=TEXT_DARK,
        fill="#F4FAF9",
        line=BORDER,
    )
    add_text_card(
        slide,
        Cm(0.9),
        Cm(9.3),
        Cm(30.2),
        Cm(4.8),
        "建议优先级：先做标签重构/浓度回归 -> 接入物理后验约束 -> 再推进软件集成。这样可以避免将不稳定研究原型过早产品化。",
        font_size=13.8,
        color=TEXT_DARK,
        fill="#ECF3FB",
        line=BORDER,
    )

    slide = add_content_slide(prs, "结论与下一阶段计划", 9)
    add_bullets_card(
        slide,
        Cm(0.8),
        Cm(3.1),
        Cm(10.1),
        Cm(5.8),
        "本次结论",
        [
            "模型 A-B-C-D 闭环已明确。",
            "Stage A 全部完成并通过验证。",
            "Stage B 核心流程跑通并有实测结果。",
            "真实 few-shot 已有信号，但表现仍需提升。",
        ],
    )
    add_bullets_card(
        slide,
        Cm(11.3),
        Cm(3.1),
        Cm(9.9),
        Cm(5.8),
        "下一步计划",
        [
            "重构任务定义：并类方案或浓度回归方案。",
            "引入物理规则判定层和可信度输出。",
            "在稳定后推进软件集成与自动报告。",
        ],
        fill="#F4FAF9",
    )
    add_text_card(
        slide,
        Cm(21.6),
        Cm(3.1),
        Cm(9.5),
        Cm(5.8),
        "答辩式一句话总结\n\n项目已从“概念验证”进入“可复现实验阶段”；下一步要解决的是“真实域可靠性”，而不是“继续扩展功能清单”。",
        font_size=13.0,
        color=TEXT_DARK,
        fill="#FFF7EE",
        line=BORDER,
    )
    add_text_card(
        slide,
        Cm(0.8),
        Cm(10.1),
        Cm(30.2),
        Cm(3.9),
        "可复用性说明：本报告由脚本自动生成，可直接迭代为组会版、开题版与答辩版。",
        font_size=14.0,
        color=TEXT_DARK,
        fill="#ECF3FB",
        line=BORDER,
    )

    prs.save(str(PPT_OUTPUT))


def main() -> None:
    assets = generate_assets()
    build_presentation(assets)
    print(f"Template: {PPT_TEMPLATE}")
    print(f"Assets: {OUTPUT_ROOT}")
    print(f"Saved PPT: {PPT_OUTPUT}")


if __name__ == "__main__":
    main()
