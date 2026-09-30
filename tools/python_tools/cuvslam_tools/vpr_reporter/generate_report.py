# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""HTML and optional PDF report generation for cuvslam_vpr_reporter runs."""

import json
import os
from dataclasses import asdict
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from cuvslam_tools.reporter.generate_report import _git_source_metadata, image_to_base64
from cuvslam_tools.vpr_reporter.execution import KNOWN_VPR_MODES

ALIGNMENT_NOTE = (
    "Map and query ground truth are read as absolute KITTI poses in one common frame and are compared "
    "directly; no trajectory alignment is applied."
)

HOW_TO_READ_NOTE = (
    "The map sequence is tracked first and every frame of it is offered to the place recognition map, which is "
    "then saved to disk. The query sequence is replayed afterwards in a fresh session that starts from a zero "
    "pose with nothing but that saved map loaded. Each queried frame asks the map where it was taken; the answer "
    "names a map frame, and the query is scored by the distance between the ground truth position of that map "
    "frame and the ground truth position of the query frame. Within {radius:.2f} m the query is a success, "
    "further away it is a false positive, and a frame the backend answers nothing for is unrecognized. In the "
    "plots the query trajectory is a ribbon: dark green where the frame was recognized correctly, amber where it "
    "was matched to the wrong place, red where it was not recognized; the thin grey line is the map sequence."
)

# A loaded place recognition map is read only (VprMap::ReadOnly), so the query session can no longer add its own
# keyframes to it and self matches are impossible. They are still counted, and reported here when they happen,
# because a non-zero count means that guarantee broke and every rate in the table is measuring the wrong thing.
SELF_MATCH_WARNING = (
    "{count} query frames matched a keyframe of the query session instead of the loaded map. A loaded map is "
    "read only, so this should be impossible; the rates below are not trustworthy until it is explained."
)

STRIP_CAPTION = (
    "Frames of the query sequence, evenly spaced over the frames it was asked about. A green border means that "
    "frame was recognized correctly, a red one that the robot did not know where it was: either nothing was "
    "recognized, or the place it matched is further than {radius:.2f} m away."
)


def save_stats_to_json(stats, output_dir):
    """Save all VPR stats to a single JSON file."""
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # The per-frame errors are what the median and mean summarize; dropping them keeps the file small.
    stats_list = [{key: value for key, value in asdict(stat).items() if key != "errors_m"} for stat in stats]

    json_file = os.path.join(stats_dir, "all_vpr_stats.json")
    with open(json_file, 'w') as f:
        json.dump(stats_list, f, indent=2)

    print(f"Saved statistics for {len(stats_list)} mode/pair combinations to {json_file}")


def calc_summary(title, stats):
    """Aggregate every (mode, pair) row into the report's total row.

    Rates and precision are recomputed from the pooled counts rather than averaged over rows, so a
    pair with more query frames weighs more, and the median error is taken over the pooled errors.
    """
    n_query_frames = sum(s.n_query_frames for s in stats)
    n_true_positives = sum(s.n_true_positives for s in stats)
    n_false_positives = sum(s.n_false_positives for s in stats)
    n_unrecognized = sum(s.n_unrecognized for s in stats)
    n_self_matches = sum(s.n_self_matches for s in stats)
    recognized = n_true_positives + n_false_positives
    errors = sorted(error for s in stats for error in s.errors_m)
    weighted_score = sum(s.mean_score * (s.n_true_positives + s.n_false_positives) for s in stats)
    # Every mean is pooled by the count it was averaged over in its own row, which is not the same
    # count for all three: weighting by anything else silently reports a number no row measured.
    n_map_frames = sum(s.n_map_frames for s in stats)
    n_queries = sum(s.n_queries for s in stats)
    map_build_time_s = sum(s.map_build_time_s for s in stats)
    add_time_ms = sum(s.mean_add_time_ms * s.n_map_frames for s in stats)
    search_time_ms = sum(s.mean_search_time_ms * s.n_queries for s in stats)

    return {
        "title": title,
        "n_query_frames": n_query_frames,
        "n_map_frames": n_map_frames,
        "n_queries": n_queries,
        "n_query_errors": sum(s.n_query_errors for s in stats),
        "n_skipped": sum(s.n_skipped for s in stats),
        "n_self_matches": n_self_matches,
        "success_rate": 100.0 * n_true_positives / n_query_frames if n_query_frames else 0.0,
        "false_positive_rate": 100.0 * n_false_positives / n_query_frames if n_query_frames else 0.0,
        "unrecognized_rate": 100.0 * n_unrecognized / n_query_frames if n_query_frames else 0.0,
        "self_match_rate": 100.0 * n_self_matches / n_query_frames if n_query_frames else 0.0,
        "precision": n_true_positives / recognized if recognized else 0.0,
        "median_error_m": errors[len(errors) // 2] if errors else 0.0,
        "mean_error_m": sum(errors) / len(errors) if errors else 0.0,
        "mean_score": weighted_score / recognized if recognized else 0.0,
        "map_build_time_s": map_build_time_s,
        "map_build_time_ms_per_frame": 1e3 * map_build_time_s / n_map_frames if n_map_frames else 0.0,
        "mean_add_time_ms": add_time_ms / n_map_frames if n_map_frames else 0.0,
        "mean_search_time_ms": search_time_ms / n_queries if n_queries else 0.0,
    }


def _image_source(path, test_folder, embed_image):
    """Reference one image by base64 payload for the PDF, or by a path the HTML can load."""
    if embed_image:
        return image_to_base64(path) if path else ""
    if path and os.path.exists(path):
        try:
            return os.path.relpath(path, test_folder)
        except ValueError:
            return path
    return ""


def _stat_row(stat, test_folder=None, embed_image=False, leader=False):
    """Build one template row, referencing its images by relative path or by base64 payload."""
    row = {key: value for key, value in asdict(stat).items() if key != "errors_m"}
    row["plot_image"] = _image_source(stat.plot_path, test_folder, embed_image)
    row["strip_image"] = _image_source(stat.strip_path, test_folder, embed_image)
    row["is_leader"] = leader
    return row


def _report_order(stats):
    """Stats grouped by sequence pair, and within a pair ordered by backend.

    The run itself is ordered the other way round - one backend over every pair, then the next -
    because a backend's map is built once and queried once. A reader wants the opposite: the pairs
    are the experiments and the backends are what is being compared within each of them.
    """
    pairs = []
    for stat in stats:
        if stat.title not in pairs:
            pairs.append(stat.title)  # first appearance, so the config's own order is kept
    modes = list(KNOWN_VPR_MODES)

    def key(stat):
        mode = modes.index(stat.vpr_mode) if stat.vpr_mode in modes else len(modes)
        return pairs.index(stat.title), mode, stat.vpr_mode

    return sorted(stats, key=key)


def _leaders(stats):
    """The best backend for each sequence pair, by success rate and then by fewest false positives.

    A report compares backends on several pairs at once, and which one won on which pair is the
    question a reader actually has; picking it out by eye from a dozen rows is what the highlight
    saves. Ties keep the first row, so the order the run was configured in decides them.
    """
    best = {}
    for index, stat in enumerate(stats):
        key = stat.title
        score = (stat.success_rate, -stat.false_positive_rate)
        if key not in best or score > best[key][0]:
            best[key] = (score, index)
    return {index for _, index in best.values()}


def generate_report(test_folder, comments, stats, generate_pdf=False, config_name=None, success_radius_m=10.0):
    """Generate an HTML VPR report and optionally render a PDF copy."""
    commit_sha, branch_name, commit_ts, provenance_warning = _git_source_metadata(
        os.path.dirname(os.path.abspath(__file__))
    )

    date_time = datetime.now().replace(microsecond=0).astimezone().isoformat('/')

    os.makedirs(test_folder, exist_ok=True)

    report_basename = f"report_{config_name}" if config_name else "report_vpr"

    template_dir = os.path.join(os.path.dirname(__file__), 'report_templates')
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

    total = calc_summary("total", stats)
    context = dict(
        date=date_time,
        branch_name=branch_name,
        commit_sha=commit_sha,
        commit_ts=commit_ts,
        provenance_warning=provenance_warning,
        comments=comments,
        success_radius_m=success_radius_m,
        alignment_note=ALIGNMENT_NOTE,
        how_to_read_note=HOW_TO_READ_NOTE.format(radius=success_radius_m),
        strip_caption=STRIP_CAPTION.format(radius=success_radius_m),
        self_match_warning=(SELF_MATCH_WARNING.format(count=total["n_self_matches"])
                            if total["n_self_matches"] else ""),
        total=total,
    )

    template = env.get_template("report.html")
    stats = _report_order(stats)
    leaders = _leaders(stats)
    html = template.render(
        stats=[_stat_row(s, test_folder=test_folder, leader=i in leaders) for i, s in enumerate(stats)], **context)

    html_file_name = os.path.join(test_folder, f"{report_basename}.html")
    with open(html_file_name, "wt") as f:
        f.write(html)

    print(f"{html_file_name} was done")

    if generate_pdf:
        try:
            from weasyprint import HTML

            pdf_template = env.get_template("report_pdf.html")
            pdf_html = pdf_template.render(
                stats=[_stat_row(s, embed_image=True, leader=i in leaders) for i, s in enumerate(stats)], **context)

            pdf_file_name = os.path.join(test_folder, f"{report_basename}.pdf")
            HTML(string=pdf_html).write_pdf(pdf_file_name)
            print(f"{pdf_file_name} was done")
        except ImportError as exc:
            raise RuntimeError(
                "PDF generation requires the optional PDF dependencies; install cuvslam-tools[pdf]."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"PDF generation failed: {exc}") from exc
