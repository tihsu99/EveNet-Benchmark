"""Scientific checkpoint profiles with an independently sourced paper reference."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import yaml


DEFAULT_REFERENCE = Path(__file__).resolve().parents[1] / "configs/paper_reference.yaml"
INK, MUTED, RULE = "#111111", "#555555", "#D5D5D5"
COLORS = ["#0072B2", "#D55E00", "#E69F00", "#3E5F78", "#56B4E9"]
MARKERS = ["o", "s", "^", "v", "P"]
IMPROVED, REGRESSED = "#187344", "#B23A3A"


def delta_style(value, reference, direction):
    if value is None or reference is None:
        return "N/A", MUTED, "unavailable"
    delta = value - reference
    improvement = -delta if direction == "min" else delta
    state = ("neutral" if direction not in ("min", "max") else
             "unchanged" if delta == 0 else "improved" if improvement > 0 else "regressed")
    return f"Δ {delta:+.3g}", {"improved": IMPROVED, "regressed": REGRESSED}.get(state, MUTED), state


def task_groups(tasks):
    groups = {}
    for name, task in tasks.items():
        groups.setdefault(task.get("group", name), []).append((name, task))
    for cases in groups.values():
        fields = cases[0][1]["evaluation"]["display"]
        if any(t["evaluation"]["display"] != fields for _, t in cases):
            raise ValueError("Grouped variants must have identical display metrics, units and directions")
    return groups


def improvement_percent(value, reference, direction):
    if not finite(value) or not finite(reference) or reference == 0 or direction not in ("min", "max"):
        return None
    result = (-1 if direction == "min" else 1) * (value - reference) / abs(reference) * 100
    if result == 0:
        return 0.0
    return result if finite(result) else None


def compact_number(value, style):
    if value is None:
        return "N/A"
    digits = int(style.get("table_significant_digits", 4))
    text = f"{value:.{digits}g}"
    if len(text) > int(style.get("table_number_max_chars", 7)):
        text = f"{value:.{max(0, digits - 2)}e}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent):+d}"
    return text


def improvement_text(value, reference, direction, style):
    percent = improvement_percent(value, reference, direction)
    if percent is None:
        return "(N/A)", MUTED
    decimals = int(style.get("improvement_decimals", 1))
    text = f"{percent:+.{decimals}f}"
    if len(text) > 8:
        text = f"{percent:+.2e}"
    return f"({text}%)", IMPROVED if percent > 0 else REGRESSED if percent < 0 else MUTED


def compare_reports(reports):
    if not 1 <= len(reports) <= 4:
        raise ValueError("Compare one to four checkpoints per report; split larger comparisons into batches")
    ids = {report.get("protocol_id") for report in reports}
    if None in ids or len(ids) != 1:
        raise ValueError("Reports need the same verified protocol_id; do not rank different datasets or budgets")
    names = [report["name"] for report in reports]
    if len(names) != len(set(names)):
        raise ValueError("Report names must be unique")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric(report, task, key):
    result = report.get("benchmarks", {}).get(task, {})
    value = result.get("evaluation", {}).get("metrics", {}).get(key)
    return value if result.get("status") == "complete" and finite(value) else None


def load_reference(path):
    reference = yaml.safe_load(Path(path).read_text())
    if not isinstance(reference, dict) or reference.get("schema_version") != 1:
        raise ValueError("Reference needs schema_version: 1")
    if not reference.get("label"):
        raise ValueError("Reference needs a label")
    for name, task in reference.get("benchmarks", {}).items():
        for key, value in task.get("metrics", {}).items():
            if value is not None and not finite(value):
                raise ValueError(f"Invalid reference value: {name}.{key}")
            if value is not None and (not task.get("source") or not task.get("context")):
                raise ValueError(f"Reference {name} needs source and context for numeric values")
        for key, interval in task.get("intervals", {}).items():
            center = task.get("metrics", {}).get(key)
            low, high = interval.get("low"), interval.get("high")
            if not all(finite(v) for v in (center, low, high)) or not low <= center <= high:
                raise ValueError(f"Invalid reference interval: {name}.{key}")
            if not interval.get("label"):
                raise ValueError(f"Reference interval {name}.{key} needs its statistical meaning in label")
    return reference


def number(value):
    return "N/A" if value is None else f"{value:.4g}"


def field_label(field):
    arrow = {"min": "↓", "max": "↑", "none": ""}[field.get("direction", "none")]
    unit = f" [{field['unit']}]" if field.get("unit") else ""
    return f"{field['label']}{unit} {arrow}".strip()


def resolve_font(preferences):
    """Use installed journal fonts when available, with portable fallbacks."""
    if isinstance(preferences, str):
        preferences = [preferences]
    available = {font.name for font in font_manager.fontManager.ttflist}
    return next((font for font in preferences if font in available), "DejaVu Sans")


def table_rules(figure, bounds, row_height, rows, style):
    """NMI Table 1: warm grey header/stripes, horizontal rules, no vertical rules."""
    left, top, width, height = bounds
    for index in range(rows + 1):
        y = 1 - (top + (index + 1) * row_height) / height
        color = style.get("table_header_color", "#D8D7C3") if index == 0 else (
            style.get("table_row_color", "#ECEBE1") if index % 2 == 0 else "white")
        figure.add_artist(Rectangle((left, y), width, row_height / height,
                                   transform=figure.transFigure, facecolor=color, edgecolor="none", zorder=0))
    for index in range(rows + 2):
        y = 1 - (top + index * row_height) / height
        figure.add_artist(Line2D([left, left + width], [y, y], transform=figure.transFigure,
                                color=INK, lw=float(style.get("table_rule_width", .55)), zorder=1))


def draw_profile(reports, tasks, reference, focus, output):
    """Compare model and head variants on shared within-metric scales."""
    groups = task_groups(tasks)
    style = reference.get('plot_style', {})
    scale = float(style.get('font_scale', 1.12))
    n = len(reports)
    heights = [max(2.9, 1.50 + .37 * (len(cases) * (n + 1) + .7 * (len(cases) - 1)))
               for cases in groups.values()]
    row_height = .46
    widths = [*style.get('summary_label_widths', [3.0, 3.4]),
              float(style.get('summary_reference_width', 2.8)),
              *([float(style.get('summary_checkpoint_width', 2.4))] * n)]
    body_top = 2.05 + row_height * (len(tasks) + 1)
    height = body_top + sum(heights) + 1.05
    width = max(float(style.get('width', 18)), sum(widths) / .92)
    figure = plt.figure(figsize=(width, height), facecolor='white')

    def label(x, inches, value, size=11, **kwargs):
        return figure.text(x, 1 - inches / height, value, fontsize=size * scale,
                           color=kwargs.pop('color', INK), **kwargs)

    def primary(name, task):
        fields = task['evaluation']['display']
        key = reference.get('benchmarks', {}).get(name, {}).get('headline')
        return next((f for f in fields if f.get('headline') or f['key'] == key), fields[0])

    label(.045, .36, 'EveNet benchmark', size=12, weight='bold')
    if any(r.get('evidence') == 'synthetic' for r in [*reports, reference]):
        label(.965, .36, 'Synthetic candidates (layout demonstration)', size=10, ha='right')
    figure.add_artist(Line2D([.045, .965], [1 - .47 / height] * 2,
                            transform=figure.transFigure, color=INK, lw=.6))
    label(.045, .94, focus if n == 1 else 'Checkpoint comparison', size=27,
          weight='bold', fontfamily=resolve_font(style.get('title_font_family', ['Harding', 'Times New Roman', 'DejaVu Serif'])))
    label(.045, 1.23, f'Fixed-budget adaptation  /  {len(groups)} benchmarks  /  '
          f"{sum(len(t['evaluation']['display']) for t in tasks.values())} metrics", size=12)
    label(.965, .92, reference['label'], size=14, ha='right', weight='bold')
    label(.965, 1.23, 'External reference' if reference.get('external', True)
          else 'Matched-protocol reference', size=10, ha='right')
    label(.045, 1.58, 'Table 1 | Primary metrics by checkpoint', size=12, weight='bold')
    table_top = 1.76
    table_rules(figure, (.045, table_top, .92, height), row_height, len(tasks), style)
    edges = [.045]
    for column_width in widths:
        edges.append(edges[-1] + .92 * column_width / sum(widths))
    centers = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    headings = ['Benchmark / head', 'Metric', 'Reference [interval]', *[r['name'] for r in reports]]
    for column, heading in enumerate(headings):
        label(edges[column] + .006 if column < 2 else centers[column], table_top + row_height * .5,
              textwrap.fill(heading, 24), size=11.5, weight='bold',
              ha='left' if column < 2 else 'center', va='center')
    for index, (name, task) in enumerate(tasks.items()):
        field = primary(name, task)
        ref_task = reference.get('benchmarks', {}).get(name, {})
        ref = ref_task.get('metrics', {}).get(field['key'])
        band = ref_task.get('intervals', {}).get(field['key'])
        ref_text = compact_number(ref, style) + (f" [{compact_number(band['low'], style)}, {compact_number(band['high'], style)}]" if band else '')
        title = task.get('label', name)
        if not task.get('variant_label'):
            title = title.split(' / ')[0]
        y = table_top + row_height * (index + 1.5)
        label(edges[0] + .006, y, title, size=11.5, va='center')
        label(edges[1] + .006, y, field_label(field), size=11.5, va='center')
        label(centers[2], y, ref_text, size=11, ha='center', va='center')
        for column, report in enumerate(reports, start=3):
            value = metric(report, name, field['key'])
            change, color = improvement_text(value, ref, field.get('direction', 'none'), style)
            props = {'size': 11.5 * scale, 'family': plt.rcParams['font.family']}
            cell = HPacker(children=[TextArea(compact_number(value, style), textprops={**props, 'color': INK}),
                                    TextArea(change, textprops={**props, 'color': color})],
                           align='baseline', pad=0, sep=4)
            figure.add_artist(AnnotationBbox(cell, (centers[column], 1 - y / height),
                                            xycoords=figure.transFigure, frameon=False, pad=0))

    top = body_top
    for group_index, (cases, section_height) in enumerate(zip(groups.values(), heights)):
        task = cases[0][1]
        fields = task['evaluation']['display']
        group_label = task.get('group_label', task.get('label', cases[0][0]))
        label(.045, top, f'{chr(97 + group_index)}   {group_label}', size=15, weight='bold')
        data = task.get('data', {})
        context = []
        if 'train_fraction' in data:
            context.append(f"{100 * data['train_fraction']:g}% training subset")
        if 'fold' in data:
            context.append(f"fold {data['fold']['index'] + 1}/{data['fold']['count']}")
        if len(cases) > 1:
            context.append('shared events and budget')
        label(.965, top, '  /  '.join(context), size=10, ha='right')
        total_rows = len(cases) * (n + 1) + .7 * (len(cases) - 1)
        step = .77 / len(fields)
        plot_top, plot_bottom = top + .68, top + section_height - .92
        for column, field in enumerate(fields):
            axis = figure.add_axes([.18 + step * column, 1 - plot_bottom / height,
                                    step * .67, (plot_bottom - plot_top) / height])
            all_values, ticks, labels = [], [], []
            for case_index, (name, case) in enumerate(cases):
                start = case_index * (n + 1 + .7)
                ref_task = reference.get('benchmarks', {}).get(name, {})
                ref = ref_task.get('metrics', {}).get(field['key'])
                band = ref_task.get('intervals', {}).get(field['key'])
                values = [ref] + [metric(r, name, field['key']) for r in reports]
                all_values.extend(v for v in values if v is not None)
                if band:
                    all_values.extend([band['low'], band['high']])
                    if len(cases) == 1:
                        axis.axvspan(band['low'], band['high'], color='#E5EAF0', zorder=0)
                    else:
                        axis.fill_betweenx([start - .4, start + n + .4], band['low'], band['high'],
                                           color='#E5EAF0', zorder=0)
                if ref is not None:
                    axis.plot([ref, ref], [start - .4, start + n + .4], color=MUTED,
                              ls=style.get('head_linestyles', {}).get(case.get('variant_label'), '--'), lw=1.5, zorder=1)
                for row, value in enumerate(values):
                    y = start + row
                    model_style = {} if row == 0 else style.get('model_styles', {}).get(reports[row - 1]['name'], {})
                    color = '#9A9A9A' if row == 0 else model_style.get('color', COLORS[(row - 1) % len(COLORS)])
                    marker = 'D' if row == 0 else model_style.get('marker', MARKERS[(row - 1) % len(MARKERS)])
                    if value is not None:
                        axis.plot(value, y, marker=marker, color=color, ms=7.5,
                                  markerfacecolor='white' if row == 0 else color, markeredgecolor='black', markeredgewidth=.7, zorder=3)
                    axis.text(1.04, y, number(value),
                              transform=axis.get_yaxis_transform(), va='center',
                              fontsize=11 * scale, color=color)
                    ticks.append(y)
                    if row == 0:
                        labels.append(f"{case['variant_label']} / Ref." if len(cases) > 1 else 'Reference')
                    else:
                        report = reports[row - 1]
                        status = report['benchmarks'].get(name, {}).get('status', 'missing')
                        labels.append(report['name'] + ('' if status == 'complete' else f' [{status}]'))
                if case_index:
                    axis.axhline(start - .85, color=RULE, lw=.8)
            if all_values:
                low, high = min(all_values), max(all_values)
                pad = max((high - low) * .2, abs(high) * .025, 1e-8)
                axis.set_xlim(low - pad, high + pad)
            else:
                axis.set_xlim(0, 1)
            axis.set_ylim(total_rows - .5, -.5)
            axis.set_yticks(ticks)
            axis.set_yticklabels([textwrap.fill(v, 22) for v in labels] if column == 0 else [], fontsize=10 * scale)
            axis.set_title(textwrap.fill(field_label(field), max(20, 100 // len(fields))),
                           loc='left', fontsize=11.5 * scale, pad=12, color=INK)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-3, 4))
            axis.xaxis.set_major_formatter(formatter)
            axis.xaxis.get_offset_text().set_fontsize(9 * scale)
            axis.tick_params(axis='x', labelsize=11 * scale, colors=INK, direction='out', length=5, width=1.5)
            axis.tick_params(axis='y', length=0, pad=10)
            axis.grid(False)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.5)
            if not all_values:
                axis.set_xticks([])
        notes = list(dict.fromkeys(reference.get('benchmarks', {}).get(name, {}).get('plot_context',
                     'No reference values supplied.') for name, _ in cases))
        label(.045, top + section_height - .34, textwrap.fill('Reference: ' + ' '.join(notes), 145), size=9)
        top += section_height
    label(.045, top + .18, 'Table: value (relative improvement %). Green: better; red: worse. ↑ (value − ref)/|ref|; ↓ (ref − value)/|ref|.', size=10)
    label(.045, top + .44, 'Shading: stated reference interval. Lower panels show raw values only. Relative improvement is N/A for zero or missing references.', size=10)
    label(.045, top + .70, 'Shared scales within each head comparison; separate scales across metrics. External paper budgets differ. No candidate seed CIs.', size=9)
    label(.045, top + .94, f"Protocol {reports[0]['protocol_id'][:24]}  /  Source details: reference.yaml and comparison.csv", size=9)
    for suffix in ('png', 'pdf', 'svg'):
        figure.savefig(output / f'comparison.{suffix}', dpi=180, facecolor='white')
    plt.close(figure)


def render(paths, output, baseline=None, reference=None, focus=None):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    compare_reports(reports)
    names = [r["name"] for r in reports]
    focus = focus or names[-1]
    if focus not in names or (baseline is not None and baseline not in names):
        raise ValueError("Unknown focus or baseline checkpoint")
    if baseline and reference:
        raise ValueError("Choose either a checkpoint baseline or an external reference")
    tasks = {name: task for name, task in reports[0]["suite"]["benchmarks"].items()
             if task.get("enabled", True) and task.get("evaluation", {}).get("display")}
    if not tasks:
        raise ValueError("No enabled benchmarks with display metrics")
    if baseline:
        base = reports[names.index(baseline)]
        ref = {"schema_version": 1, "label": baseline, "external": False, "benchmarks": {
            name: {"metrics": {f["key"]: metric(base, name, f["key"]) for f in task["evaluation"]["display"]},
                   "context": "Checkpoint evaluated with the same probe protocol.", "source": baseline}
            for name, task in tasks.items()}}
    else:
        ref = load_reference(reference or DEFAULT_REFERENCE)
        ref["external"] = True
    order = ref.get("benchmark_order", [])
    tasks = {name: tasks[name] for name in dict.fromkeys([*order, *tasks]) if name in tasks}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, task in tasks.items():
        ref_task = ref.get("benchmarks", {}).get(name, {})
        for report in reports:
            result = report["benchmarks"].get(name, {"status": "missing"})
            for field in task["evaluation"]["display"]:
                value = metric(report, name, field["key"])
                reference_value = ref_task.get("metrics", {}).get(field["key"])
                interval = ref_task.get("intervals", {}).get(field["key"], {})
                rows.append({"checkpoint": report["name"], "benchmark": name, "metric": field["key"],
                             "value": value, "reference": ref["label"], "reference_value": reference_value,
                             "delta_to_baseline": None if value is None or reference_value is None else value - reference_value,
                             "improvement_percent": improvement_percent(value, reference_value, field.get("direction", "none")),
                             "direction": field.get("direction", "none"), "status": result["status"],
                             "delta_assessment": delta_style(value, reference_value, field.get("direction", "none"))[2],
                             "variant": task.get("variant_label", ""),
                             "reference_low": interval.get("low"), "reference_high": interval.get("high"),
                             "reference_interval": interval.get("label", "unavailable"),
                             "source": ref_task.get("source", ""), "reference_context": ref_task.get("context", ""),
                             "note": result.get("error") or "; ".join(result.get("evaluation", {}).get("notes", []))})
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "reference.yaml").write_text(yaml.safe_dump(ref, sort_keys=False))
    font = resolve_font(ref.get("plot_style", {}).get("font_family", ["Graphik Naturel", "Arial", "DejaVu Sans"]))
    with plt.rc_context({"font.family": font, "pdf.fonttype": 42, "ps.fonttype": 42}):
        draw_profile(reports, tasks, ref, focus, output)
    # A compact model-by-metric table; full provenance remains in CSV and YAML.
    escape = lambda value: html.escape("N/A" if value is None else str(value))
    style = ref.get("plot_style", {})
    document = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>EveNet report data</title>",
                "<style>body{font:16px Arial,sans-serif;margin:40px;color:#111}h1{font-family:'Times New Roman',serif}"
                "table{border-collapse:collapse;min-width:100%;font-size:14px}th,td{border:0;border-bottom:.6px solid #111;"
                "padding:8px 12px;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}"
                "th{background:var(--header);border-top:.6px solid #111}tbody tr:nth-child(even){background:var(--row)}"
                ".scroll{overflow-x:auto}small{color:#555}h2{font-size:18px}</style>",
                f"<body style='--header:{escape(style.get('table_header_color', '#D8D7C3'))};"
                f"--row:{escape(style.get('table_row_color', '#ECEBE1'))}'>",
                "<h1>EveNet report: data and provenance</h1>",
                "<p><a href='comparison.pdf'>Scientific profile (PDF)</a> · <a href='comparison.png'>PNG</a> · "
                "<a href='comparison.svg'>SVG</a> · <a href='comparison.csv'>CSV</a> · <a href='reference.yaml'>Reference</a></p>"]
    if any(r.get("evidence") == "synthetic" for r in [*reports, ref]):
        document.append("<h2>SYNTHETIC DEMONSTRATION — NOT PHYSICS RESULTS</h2>")
    document.append("<p>External paper results are context, not a matched-budget ablation. "
                    "Missing or nonfinite metrics are N/A. Shading uses only explicitly sourced reference intervals; "
                    "Cells show value (relative improvement %), direction-adjusted so green means better. "
                    "The percentage is unavailable for zero/missing references or unspecified direction.</p><div class='scroll'><table><thead><tr>")
    document.extend(f"<th>{escape(key)}</th>" for key in ["Benchmark / head", "Metric", ref["label"] + " [interval]", *names])
    document.append("</tr></thead><tbody>")
    for name, task in tasks.items():
        ref_task = ref.get("benchmarks", {}).get(name, {})
        for field in task["evaluation"]["display"]:
            cells = [escape(task.get("label", name)), escape(field_label(field))]
            band = ref_task.get("intervals", {}).get(field["key"])
            ref_value = ref_task.get("metrics", {}).get(field["key"])
            ref_text = compact_number(ref_value, style)
            if band:
                ref_text += f" [{compact_number(band['low'], style)}, {compact_number(band['high'], style)}]"
            cells.append(escape(ref_text))
            for report in reports:
                result = report.get("benchmarks", {}).get(name, {})
                value = metric(report, name, field["key"])
                change, color = improvement_text(value, ref_value, field.get("direction", "none"), style)
                cells.append(f"{escape(compact_number(value, style))} <span style='color:{color}'>{escape(change)}</span>"
                             if result.get("status") == "complete" else f"N/A ({escape(result.get('status', 'missing'))})")
            document.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    document.append("</tbody></table></div><h2>Reference definitions and provenance</h2>")
    for name, task in tasks.items():
        source = ref.get("benchmarks", {}).get(name, {})
        document.append(f"<p><b>{escape(task.get('label', name))}</b><br>{escape(source.get('context', 'No reference supplied.'))}"
                        f"<br><small>Source: {escape(source.get('source'))}</small></p>")
    document.append("</body></html>")
    (output / "report.html").write_text("\n".join(document))
    return output / "report.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline", help="Optional matched-protocol checkpoint baseline")
    group.add_argument("--reference", help=f"External YAML reference (default: {DEFAULT_REFERENCE})")
    parser.add_argument("--focus", help="Legacy accepted option; multi-checkpoint tables always show all inputs")
    args = parser.parse_args()
    print(render(args.reports, args.output, args.baseline, args.reference, args.focus))


if __name__ == "__main__":
    main()
