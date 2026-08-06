from pathlib import Path
from math import erfc, sqrt
import re
from string import ascii_uppercase

import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas


BASE_DIR = Path(r"C:\satd_microservice\classification")
OUT_DIR = BASE_DIR / "results" / "LLMs" / "size_effect"

B_BOOT = 500
RANDOM_SEED = 42
ALPHA = 0.05
MIN_CLIFF_DELTA = 0.147

CONTEXTS = {
    "few_shot_file_content": {
        "title": "Few-shot with file-content context",
        "folder": BASE_DIR / "results" / "LLMs" / "with_context" / "few-shot" / "file_content" / "new",
        "svg": OUT_DIR / "reduced_sk_boxplots_few_shot_file_content.svg",
        "xlsx": OUT_DIR / "reduced_sk_boxplots_few_shot_file_content.xlsx",
        "pdf_dir": OUT_DIR / "reduced_sk_subfigures_few_shot_file_content",
    },
    "zero_shot_window": {
        "title": "Zero-shot with window context",
        "folder": BASE_DIR / "results" / "LLMs" / "with_context" / "zero_shot" / "Window",
        "svg": OUT_DIR / "reduced_sk_boxplots_zero_shot_window.svg",
        "xlsx": OUT_DIR / "reduced_sk_boxplots_zero_shot_window.xlsx",
        "pdf_dir": OUT_DIR / "reduced_sk_subfigures_zero_shot_window",
    },
}

CATEGORY_ORDER = [
    "code",
    "compatibility/dependency",
    "configuration",
    "database access",
    "defect",
    "design",
    "infrastructure and pipeline",
    "requirement",
    "security",
    "service communication",
    "service design",
    "service operations and deployment",
    "test",
]

CATEGORY_DISPLAY = {
    "infrastructure and pipeline": "infrastructure/pipeline",
    "service operations and deployment": "service ops./deployment",
}

MODEL_DISPLAY = {
    "claude": "Claude-Sonnet-4.5",
    "deepseek": "DeepSeek-R1",
    "gemini": "Gemini-3-Flash",
    "gemma": "Gemma-4-26B",
    "gpt": "GPT-5-mini",
    "llama": "LLaMA-3.3-70B",
    "phi": "Phi-4",
    "qwen": "Qwen3-Coder-Next",
}

MODEL_SHORT = {
    "Claude-Sonnet-4.5": "Claude-Sonnet-4.5",
    "DeepSeek-R1": "DeepSeek-R1",
    "Gemini-3-Flash": "Gemini-3-Flash",
    "Gemma-4-26B": "Gemma-4-26B",
    "GPT-5-mini": "GPT-5-mini",
    "LLaMA-3.3-70B": "LLaMA-3.3-70B",
    "Phi-4": "Phi-4",
    "Qwen3-Coder-Next": "Qwen3-Coder-Next",
}

MODEL_COLORS = {
    "Claude-Sonnet-4.5": "#2F5597",
    "DeepSeek-R1": "#70AD47",
    "Gemini-3-Flash": "#00A2E8",
    "Gemma-4-26B": "#A9D18E",
    "GPT-5-mini": "#C00000",
    "LLaMA-3.3-70B": "#8064A2",
    "Phi-4": "#B4A7D6",
    "Qwen3-Coder-Next": "#548235",
}


def normalize_model_name(path: Path) -> str:
    name = path.name.lower()
    suffixes = [
        "_few_shot_dynamic_with_context_content_file.xlsx",
        "-few_shot_dynamic_with_context.xlsx",
        "_few_shot_dynamic_with_context.xlsx",
        "_few_shot_dynamic.xlsx",
        "_zero_shot_with_context_window.xlsx",
        "_zero_shot_window_original_prompt.xlsx",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("deepseek-r", "deepseek").replace("deepseek-v4-pro", "deepseek").replace("phi-4", "phi")


def normalize_comment(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip().lower()


def normalize_label(value) -> str:
    label = str(value).strip().lower()
    if label in {"compatibility", "dependency"}:
        return "compatibility/dependency"
    return label


def load_predictions(folder: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(folder.glob("*.xlsx")):
        model_key = normalize_model_name(path)
        if model_key not in MODEL_DISPLAY:
            continue
        df = pd.read_excel(path, sheet_name="Predictions")
        df = df[["Comment", "True_Label", "Predicted_Label"]].copy()
        df["comment_key"] = df["Comment"].map(normalize_comment)
        df["True_Label"] = df["True_Label"].map(normalize_label)
        df["Predicted_Label"] = df["Predicted_Label"].map(normalize_label)
        df = df[df["True_Label"] != "unclear"]
        frames[model_key] = df.drop_duplicates("comment_key").sort_values("comment_key").reset_index(drop=True)
    return frames


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for label in np.unique(y_true):
        idx = np.flatnonzero(y_true == label)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def category_f1(y_true: np.ndarray, y_pred: np.ndarray, category: str) -> float:
    true_pos = np.sum((y_true == category) & (y_pred == category))
    false_pos = np.sum((y_true != category) & (y_pred == category))
    false_neg = np.sum((y_true == category) & (y_pred != category))
    denom = 2 * true_pos + false_pos + false_neg
    return 0.0 if denom == 0 else float((2 * true_pos) / denom)


def mann_whitney_p_and_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    nx = len(x)
    ny = len(y)
    pooled = pd.Series(np.concatenate([x, y]))
    ranks = pooled.rank(method="average").to_numpy()
    rank_x = ranks[:nx].sum()
    u_value = rank_x - nx * (nx + 1) / 2
    mean_u = nx * ny / 2
    std_u = sqrt(nx * ny * (nx + ny + 1) / 12)
    z_value = 0.0 if std_u == 0 else (u_value - mean_u) / std_u
    p_value = erfc(abs(z_value) / sqrt(2))
    cliffs_delta = (2 * u_value) / (nx * ny) - 1
    return float(p_value), abs(float(cliffs_delta))


def assign_groups(values_by_model: dict[str, np.ndarray]) -> tuple[dict[str, int], pd.DataFrame]:
    ordered = sorted(values_by_model, key=lambda model: values_by_model[model].mean(), reverse=True)
    assignments = {}
    diagnostics = []

    def recurse(items: list[str], next_rank: int) -> int:
        if len(items) == 1:
            assignments[items[0]] = next_rank
            return next_rank + 1

        pooled = np.concatenate([values_by_model[item] for item in items])
        best_gain = -np.inf
        best_split = None
        for split_idx in range(1, len(items)):
            left = items[:split_idx]
            right = items[split_idx:]
            left_vals = np.concatenate([values_by_model[item] for item in left])
            right_vals = np.concatenate([values_by_model[item] for item in right])
            gain = (
                len(left_vals) * (left_vals.mean() - pooled.mean()) ** 2
                + len(right_vals) * (right_vals.mean() - pooled.mean()) ** 2
            )
            if gain > best_gain:
                best_gain = gain
                best_split = (left, right)

        left, right = best_split
        left_vals = np.concatenate([values_by_model[item] for item in left])
        right_vals = np.concatenate([values_by_model[item] for item in right])
        p_value, delta = mann_whitney_p_and_delta(left_vals, right_vals)
        accepted = p_value < ALPHA and delta >= MIN_CLIFF_DELTA
        diagnostics.append(
            {
                "left_models": ", ".join(MODEL_DISPLAY[item] for item in left),
                "right_models": ", ".join(MODEL_DISPLAY[item] for item in right),
                "p_value": p_value,
                "cliffs_delta_abs": delta,
                "accepted_split": accepted,
            }
        )
        if accepted:
            next_rank = recurse(left, next_rank)
            next_rank = recurse(right, next_rank)
            return next_rank

        for item in items:
            assignments[item] = next_rank
        return next_rank + 1

    recurse(ordered, 1)
    return assignments, pd.DataFrame(diagnostics)


def build_context_data(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = load_predictions(folder)
    common_comments = set.intersection(*(set(df["comment_key"]) for df in frames.values()))
    frames = {
        model: df[df["comment_key"].isin(common_comments)].sort_values("comment_key").reset_index(drop=True)
        for model, df in frames.items()
    }
    first = next(iter(frames.values()))
    y_true = first["True_Label"].to_numpy()
    categories = [category for category in CATEGORY_ORDER if category in set(y_true)]

    rng = np.random.default_rng(RANDOM_SEED)
    boot_indices = [stratified_bootstrap_indices(y_true, rng) for _ in range(B_BOOT)]

    rows = []
    boot_rows = []
    diagnostic_parts = []
    for category in categories:
        values_by_model = {}
        for model_key, df in frames.items():
            y_pred = df["Predicted_Label"].to_numpy()
            observed_f1 = category_f1(y_true, y_pred, category)
            boot_values = np.array([category_f1(y_true[idx], y_pred[idx], category) for idx in boot_indices])
            values_by_model[model_key] = boot_values
            for sample_id, value in enumerate(boot_values):
                boot_rows.append(
                    {
                        "category": category,
                        "model": MODEL_DISPLAY[model_key],
                        "bootstrap_id": sample_id,
                        "f1_score": value,
                    }
                )
            rows.append(
                {
                    "category": category,
                    "category_display": CATEGORY_DISPLAY.get(category, category),
                    "model_key": model_key,
                    "model": MODEL_DISPLAY[model_key],
                    "observed_f1": observed_f1,
                    "bootstrap_mean_f1": float(boot_values.mean()),
                    "ci_low": float(np.quantile(boot_values, 0.025)),
                    "ci_high": float(np.quantile(boot_values, 0.975)),
                }
            )

        groups, diagnostics = assign_groups(values_by_model)
        for row in rows:
            if row["category"] == category:
                rank = groups[row["model_key"]]
                row["sk_rank"] = rank
                row["sk_group"] = ascii_uppercase[rank - 1] if rank <= len(ascii_uppercase) else f"G{rank}"
        if not diagnostics.empty:
            diagnostics.insert(0, "category", category)
            diagnostic_parts.append(diagnostics)

    ranking = pd.DataFrame(rows).sort_values(["category", "sk_rank", "observed_f1"], ascending=[True, True, False])
    boot = pd.DataFrame(boot_rows)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True) if diagnostic_parts else pd.DataFrame()
    return ranking, boot, diagnostics


def five_number(values: pd.Series) -> dict[str, float]:
    qs = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "p05": float(qs.loc[0.05]),
        "q1": float(qs.loc[0.25]),
        "median": float(qs.loc[0.5]),
        "q3": float(qs.loc[0.75]),
        "p95": float(qs.loc[0.95]),
        "mean": float(values.mean()),
    }


def write_workbook(path: Path, ranking: pd.DataFrame, boot: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="Model_Category_SK", index=False)
        boot.to_excel(writer, sheet_name="Bootstrap_F1", index=False)
        diagnostics.to_excel(writer, sheet_name="SK_Diagnostics", index=False)


def write_boxplot_svg(title: str, destination: Path, ranking: pd.DataFrame, boot: pd.DataFrame) -> None:
    categories = [category for category in CATEGORY_ORDER if category in set(ranking["category"])]
    panel_cols = 3
    panel_rows = int(np.ceil(len(categories) / panel_cols))
    panel_w = 460
    panel_h = 238
    margin_l = 58
    margin_t = 22
    margin_r = 28
    margin_b = 42
    width = margin_l + margin_r + panel_cols * panel_w
    height = margin_t + margin_b + panel_rows * panel_h
    plot_l = 48
    plot_t = 40
    plot_r = 16
    plot_b = 46
    box_w = 18

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]

    for idx, category in enumerate(categories):
        row = idx // panel_cols
        col = idx % panel_cols
        px = margin_l + col * panel_w
        py = margin_t + row * panel_h
        panel_ranking = ranking[ranking["category"] == category].sort_values(["sk_rank", "observed_f1"], ascending=[True, False])
        models = list(panel_ranking["model"])
        cat_boot = boot[boot["category"] == category]
        y_max = max(0.12, float(cat_boot["f1_score"].quantile(0.99)) + 0.05)
        y_max = min(1.0, max(y_max, float(panel_ranking["observed_f1"].max()) + 0.08))
        plot_w = panel_w - plot_l - plot_r
        plot_h = panel_h - plot_t - plot_b

        def x_for(model_index: int) -> float:
            return px + plot_l + (model_index + 0.5) * plot_w / len(models)

        def y_for(value: float) -> float:
            return py + plot_t + plot_h - (value / y_max) * plot_h

        lines.append(f'<rect x="{px + 6}" y="{py + 8}" width="{panel_w - 12}" height="{panel_h - 18}" fill="#FFFFFF" stroke="#D1D5DB" stroke-width="1"/>')

        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            if tick <= y_max + 1e-9:
                y = y_for(tick)
                lines.append(f'<line x1="{px + plot_l}" y1="{y:.1f}" x2="{px + panel_w - plot_r}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
                lines.append(f'<text x="{px + plot_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#111827">{tick:.2f}</text>')

        lines.append(f'<line x1="{px + plot_l}" y1="{py + plot_t + plot_h}" x2="{px + panel_w - plot_r}" y2="{py + plot_t + plot_h}" stroke="#374151" stroke-width="1"/>')
        lines.append(f'<line x1="{px + plot_l}" y1="{py + plot_t}" x2="{px + plot_l}" y2="{py + plot_t + plot_h}" stroke="#374151" stroke-width="1"/>')

        for model_idx, model in enumerate(models):
            values = cat_boot[cat_boot["model"] == model]["f1_score"]
            stats = five_number(values)
            x = x_for(model_idx)
            y05 = y_for(stats["p05"])
            yq1 = y_for(stats["q1"])
            ymed = y_for(stats["median"])
            yq3 = y_for(stats["q3"])
            y95 = y_for(stats["p95"])
            ymean = y_for(stats["mean"])
            color = MODEL_COLORS[model]
            group = panel_ranking[panel_ranking["model"] == model]["sk_group"].iloc[0]

            lines.append(f'<line x1="{x:.1f}" y1="{y05:.1f}" x2="{x:.1f}" y2="{y95:.1f}" stroke="#374151" stroke-width="1"/>')
            lines.append(f'<line x1="{x - box_w / 2:.1f}" y1="{y05:.1f}" x2="{x + box_w / 2:.1f}" y2="{y05:.1f}" stroke="#374151" stroke-width="1"/>')
            lines.append(f'<line x1="{x - box_w / 2:.1f}" y1="{y95:.1f}" x2="{x + box_w / 2:.1f}" y2="{y95:.1f}" stroke="#374151" stroke-width="1"/>')
            lines.append(f'<rect x="{x - box_w / 2:.1f}" y="{yq3:.1f}" width="{box_w}" height="{max(1, yq1 - yq3):.1f}" fill="{color}" fill-opacity="0.72" stroke="#374151" stroke-width="1"/>')
            lines.append(f'<line x1="{x - box_w / 2:.1f}" y1="{ymed:.1f}" x2="{x + box_w / 2:.1f}" y2="{ymed:.1f}" stroke="#111827" stroke-width="1.4"/>')
            lines.append(f'<circle cx="{x:.1f}" cy="{ymean:.1f}" r="3.6" fill="#FFFFFF" stroke="#111827" stroke-width="1.2"/>')
            lines.append(f'<text x="{x:.1f}" y="{py + plot_t - 8}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#111827">{group}</text>')

            label = MODEL_SHORT[model]
            lx = x - 2
            ly = py + plot_t + plot_h + 29
            lines.append(f'<text x="{lx:.1f}" y="{ly:.1f}" transform="rotate(-42 {lx:.1f} {ly:.1f})" text-anchor="end" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#111827">{label}</text>')

    lines.append(f'<text x="22" y="{height / 2:.1f}" transform="rotate(-90 22 {height / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">F1-score</text>')
    lines.append("</svg>")
    destination.write_text("\n".join(lines), encoding="utf-8")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def draw_rotated_text(c: canvas.Canvas, x: float, y: float, text: str, angle: float, font: str, size: float) -> None:
    c.saveState()
    c.translate(x, y)
    c.rotate(angle)
    c.setFont(font, size)
    c.drawRightString(0, 0, text)
    c.restoreState()


def write_category_pdf(category: str, destination: Path, ranking: pd.DataFrame, boot: pd.DataFrame) -> None:
    page_w = 640
    page_h = 370
    plot_l = 72
    plot_t = 68
    plot_r = 24
    plot_b = 96
    plot_w = page_w - plot_l - plot_r
    plot_h = page_h - plot_t - plot_b
    box_w = 30

    panel_ranking = ranking[ranking["category"] == category].sort_values(
        ["sk_rank", "observed_f1"], ascending=[True, False]
    )
    models = list(panel_ranking["model"])
    cat_boot = boot[boot["category"] == category]
    y_max = max(0.12, float(cat_boot["f1_score"].quantile(0.99)) + 0.05)
    y_max = min(1.0, max(y_max, float(panel_ranking["observed_f1"].max()) + 0.08))

    def x_for(model_index: int) -> float:
        return plot_l + (model_index + 0.5) * plot_w / len(models)

    def y_for(value: float) -> float:
        return plot_b + (value / y_max) * plot_h

    destination.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(destination), pagesize=(page_w, page_h))
    c.setTitle(CATEGORY_DISPLAY.get(category, category))

    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)
    c.setStrokeColorRGB(0.82, 0.84, 0.87)
    c.rect(12, 14, page_w - 24, page_h - 28, stroke=1, fill=0)

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        if tick <= y_max + 1e-9:
            y = y_for(tick)
            c.setStrokeColorRGB(0.90, 0.91, 0.93)
            c.line(plot_l, y, page_w - plot_r, y)
            c.setFillColorRGB(0.07, 0.09, 0.15)
            c.setFont("Helvetica-Bold", 13)
            c.drawRightString(plot_l - 8, y - 3, f"{tick:.2f}")

    c.setStrokeColorRGB(0.21, 0.25, 0.32)
    c.line(plot_l, plot_b, page_w - plot_r, plot_b)
    c.line(plot_l, plot_b, plot_l, plot_b + plot_h)

    for model_idx, model in enumerate(models):
        values = cat_boot[cat_boot["model"] == model]["f1_score"]
        stats = five_number(values)
        x = x_for(model_idx)
        y05 = y_for(stats["p05"])
        yq1 = y_for(stats["q1"])
        ymed = y_for(stats["median"])
        yq3 = y_for(stats["q3"])
        y95 = y_for(stats["p95"])
        ymean = y_for(stats["mean"])
        group = panel_ranking[panel_ranking["model"] == model]["sk_group"].iloc[0]
        r, g, b = hex_to_rgb(MODEL_COLORS[model])

        c.setStrokeColorRGB(0.21, 0.25, 0.32)
        c.setLineWidth(0.9)
        c.line(x, y05, x, y95)
        c.line(x - box_w / 2, y05, x + box_w / 2, y05)
        c.line(x - box_w / 2, y95, x + box_w / 2, y95)
        c.setFillColorRGB(r, g, b)
        c.rect(x - box_w / 2, min(yq1, yq3), box_w, max(1, abs(yq3 - yq1)), stroke=1, fill=1)
        c.setLineWidth(1.2)
        c.line(x - box_w / 2, ymed, x + box_w / 2, ymed)
        c.setFillColorRGB(1, 1, 1)
        c.circle(x, ymean, 3.8, stroke=1, fill=1)

        c.setFillColorRGB(0.07, 0.09, 0.15)
        c.setFont("Helvetica-Bold", 15)
        c.drawCentredString(x, plot_b + plot_h + 16, group)
        draw_rotated_text(c, x - 2, plot_b - 20, MODEL_SHORT[model], 43, "Helvetica-Bold", 13)

    c.saveState()
    c.translate(20, plot_b + plot_h / 2)
    c.rotate(90)
    c.setFillColorRGB(0.07, 0.09, 0.15)
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(0, 0, "F1-score")
    c.restoreState()
    c.showPage()
    c.save()


def write_category_pdfs(destination_dir: Path, ranking: pd.DataFrame, boot: pd.DataFrame) -> list[Path]:
    categories = [category for category in CATEGORY_ORDER if category in set(ranking["category"])]
    written = []
    for index, category in enumerate(categories, start=1):
        output = destination_dir / f"{index:02d}_{slugify(category)}.pdf"
        write_category_pdf(category, output, ranking, boot)
        written.append(output)
    return written


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for context in CONTEXTS.values():
        ranking, boot, diagnostics = build_context_data(context["folder"])
        write_workbook(context["xlsx"], ranking, boot, diagnostics)
        write_boxplot_svg(context["title"], context["svg"], ranking, boot)
        pdfs = write_category_pdfs(context["pdf_dir"], ranking, boot)
        print(f"Saved workbook: {context['xlsx']}")
        print(f"Saved figure: {context['svg']}")
        print(f"Saved {len(pdfs)} PDF subfigures: {context['pdf_dir']}")


if __name__ == "__main__":
    main()
