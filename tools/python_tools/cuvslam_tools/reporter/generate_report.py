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

"""HTML and optional PDF report generation for cuVSLAM reporter runs."""

import os
import json
import base64
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader


def save_stats_to_json(stats, output_dir):
    """Save all stats to a single JSON file."""

    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Convert stats to list of dictionaries
    stats_list = []
    for stat in stats:
        stat_dict = {
            'sequence_title': stat.sequence_title,
            'n_frames': stat.n_frames,
            'tracking_time': stat.tracking_time,
            'average_fps': stat.average_fps,
            'bird_view_with_errors_path': stat.bird_view_with_errors_path,
            'gt_av_translation_error': stat.gt_av_translation_error,
            'gt_av_rotation_error': stat.gt_av_rotation_error,
            'gt_n_error_segments': stat.gt_n_error_segments,
            'gt_simple_error': stat.gt_simple_error,
            'num_tracking_losts': stat.num_tracking_losts,
            'odometry_mode': stat.odometry_mode
        }
        stats_list.append(stat_dict)

    # Save to JSON file
    json_file = os.path.join(stats_dir, "all_stats.json")
    with open(json_file, 'w') as f:
        json.dump(stats_list, f, indent=2)

    print(f"Saved statistics for {len(stats_list)} sequences to {json_file}")


def get_fps(tracking_time, n_frames):
    """Return frames per second, or -1 when no tracking time was recorded."""
    if tracking_time > 0:
        fps = n_frames / tracking_time
        return fps
    else:
        return -1


def image_to_base64(image_path):
    """Convert an image file to base64 data URI."""
    if not os.path.exists(image_path):
        return ""

    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            base64_data = base64.b64encode(img_data).decode('utf-8')
            # Detect image format from extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
            return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        print(f"Error converting image to base64: {image_path}, {e}")
        return ""


def filter_rows(stats, suffix):
    """Filter sequence stats by a sequence-title suffix."""
    return list(filter(lambda x: x['sequence_title'].endswith(suffix), stats))


def calc_summary(title, stats):
    """Aggregate sequence statistics for the report summary section."""
    tracking_time = 0
    n_frames = 0
    total_gt_av_translation_error = 0
    total_gt_av_rotation_error = 0
    total_gt_n_error_segments = 0
    total_gt_simple_error = 0
    num_correct_sequences = 0
    num_stats_with_gt = 0
    total_tracking_losts = 0
    for s in stats:
        tracking_time += s.tracking_time
        n_frames += s.n_frames
        total_gt_av_translation_error += s.gt_av_translation_error
        total_gt_av_rotation_error += s.gt_av_rotation_error
        total_gt_n_error_segments += s.gt_n_error_segments
        total_gt_simple_error += s.gt_simple_error
        total_tracking_losts += s.num_tracking_losts
        if s.gt_av_translation_error > 0:
            num_stats_with_gt += 1
            if s.gt_av_translation_error <= 0.011:
                num_correct_sequences += 1

    summary_av_gt_translation_error = 0
    summary_av_gt_rotation_error = 0

    summary_av_gt_simple_error = 0
    if num_stats_with_gt > 0:
        summary_av_gt_translation_error = total_gt_av_translation_error / num_stats_with_gt
        summary_av_gt_rotation_error = total_gt_av_rotation_error / num_stats_with_gt
        summary_av_gt_simple_error = total_gt_simple_error / num_stats_with_gt
    return {
        "title": title,
        "n_frames": n_frames,
        "average_fps": get_fps(tracking_time, n_frames),
        "summary_av_gt_translation_error": summary_av_gt_translation_error,
        "summary_av_gt_rotation_error": summary_av_gt_rotation_error,
        "summary_av_gt_simple_error": summary_av_gt_simple_error,
        "total_sequences": num_stats_with_gt,
        "num_correct_sequences": num_correct_sequences,
        "total_tracking_losts": total_tracking_losts
    }


def _aggregate_by_length(stats) -> Tuple[List[float], List[float], List[float]]:
    """Bin per-segment error points from all sequences by path length.

    Returns (lengths, mean_t_pct, mean_r_deg_per_m) sorted by length.
    Bins with fewer than 3 samples are skipped (matches reporter's >2.5 cutoff).
    """
    t_by_len = defaultdict(list)
    r_by_len = defaultdict(list)
    for s in stats:
        for p in getattr(s, 'seg_err_points', []) or []:
            t_by_len[p['length']].append(p['t_pct'])
            r_by_len[p['length']].append(p['r_deg_per_m'])

    lengths, mean_t, mean_r = [], [], []
    for length in sorted(t_by_len):
        if len(t_by_len[length]) < 3 or not r_by_len[length]:
            continue
        lengths.append(length)
        mean_t.append(sum(t_by_len[length]) / len(t_by_len[length]))
        mean_r.append(sum(r_by_len[length]) / len(r_by_len[length]))
    return lengths, mean_t, mean_r


def _save_avg_error_plots(stats, plots_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Render cross-sequence avg translation/rotation error vs path length.

    Returns (avg_tl_path, avg_rl_path), or (None, None) if no data.
    """
    lengths, mean_t, mean_r = _aggregate_by_length(stats)
    if not lengths:
        return None, None

    os.makedirs(plots_dir, exist_ok=True)
    avg_tl_path = os.path.join(plots_dir, "avg_tl.png")
    avg_rl_path = os.path.join(plots_dir, "avg_rl.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(lengths, mean_t, marker='s', color='#0000FF', label='Translation Error')
    ax.set_xlabel('Path Length [m]')
    ax.set_ylabel('Translation Error [%]')
    ax.set_title('Average translation error vs path length')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(avg_tl_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(lengths, mean_r, marker='s', color='#0000FF', label='Rotation Error')
    ax.set_xlabel('Path Length [m]')
    ax.set_ylabel('Rotation Error [deg/m]')
    ax.set_title('Average rotation error vs path length')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(avg_rl_path)
    plt.close(fig)

    return avg_tl_path, avg_rl_path


def _git_output(args: List[str], repo_dir: str) -> Tuple[str, str]:
    """Run one git command in repo_dir and return (stdout, error_message).

    Git output is captured rather than inherited so that a failed provenance query
    cannot write to the log of whatever pipeline is running the reporter.
    """
    try:
        completed = subprocess.run(
            ['git', '-C', repo_dir] + args,
            capture_output=True,
            universal_newlines=True,
            check=True,
        )
    except FileNotFoundError:
        return "", "git is not installed"
    except subprocess.CalledProcessError as exc:
        return "", " ".join((exc.stderr or "").split()) or f"git exited with {exc.returncode}"
    return completed.stdout.strip(), ""


def _git_source_metadata(repo_dir: str) -> Tuple[str, str, str, str]:
    """Return (commit_sha, branch_name, commit_timestamp, warning) describing repo_dir.

    The warning is empty only when every field resolved. Otherwise it explains why the
    unknown ones are unknown, so the report carries the reason too; reports are published
    as an artifact of their own and are read without the run log next to them.
    """
    commit_sha, error = _git_output(['rev-parse', 'HEAD'], repo_dir)
    if not commit_sha:
        warning = f"Provenance unavailable for {repo_dir}: {error}"
        print(f"Warning: {warning}")
        return "unknown", "unknown", "unknown", warning

    branch_name, branch_error = _git_output(['rev-parse', '--abbrev-ref', 'HEAD'], repo_dir)
    commit_ts, commit_ts_error = _git_output(['show', '-s', '--format=%ci', commit_sha], repo_dir)

    warning = ""
    failures = []
    if branch_error:
        failures.append(f"branch name: {branch_error}")
    if commit_ts_error:
        failures.append(f"commit date: {commit_ts_error}")
    if failures:
        warning = f"Provenance incomplete for {repo_dir}: {'; '.join(failures)}"
        print(f"Warning: {warning}")

    return commit_sha, branch_name or "unknown", commit_ts.replace(' ', '/') or "unknown", warning


def generate_report(test_folder, comments, stats, generate_pdf=False, config_name=None):
    """Generate an HTML report and optionally render a PDF copy."""
    commit_sha, branch_name, commit_ts, provenance_warning = _git_source_metadata(
        os.path.dirname(os.path.abspath(__file__))
    )

    date_time = datetime.now().replace(microsecond=0).astimezone().isoformat('/')

    os.makedirs(test_folder, exist_ok=True)

    # Use config name if provided, otherwise use default name
    report_basename = f"report_{config_name}" if config_name else "report"

    # For HTML: reference images in existing plots folder
    stats_for_html = []
    for s in stats:
        # Calculate relative path from report to existing plot image
        image_relative_path = ""
        if s.bird_view_with_errors_path and os.path.exists(s.bird_view_with_errors_path):
            # Get relative path from test_folder to the image
            try:
                image_relative_path = os.path.relpath(s.bird_view_with_errors_path, test_folder)
            except ValueError:
                # If on different drives (Windows), use absolute path
                image_relative_path = s.bird_view_with_errors_path

        stat_dict = {
            'sequence_title': s.sequence_title,
            'n_frames': s.n_frames,
            'average_fps': s.average_fps,
            'gt_av_translation_error': s.gt_av_translation_error,
            'gt_av_rotation_error': s.gt_av_rotation_error,
            'gt_simple_error': s.gt_simple_error,
            'bird_view_with_errors_path': s.bird_view_with_errors_path,
            'bird_view_image_path': image_relative_path  # Relative path for HTML
        }
        stats_for_html.append(stat_dict)

    # Render cross-sequence error-vs-path-length plots; embedded under the
    # "Average" header above the per-sequence sections.
    plots_dir = os.path.join(test_folder, "plots")
    avg_tl_abs, avg_rl_abs = _save_avg_error_plots(stats, plots_dir)
    avg_tl_rel = os.path.relpath(avg_tl_abs, test_folder) if avg_tl_abs else ""
    avg_rl_rel = os.path.relpath(avg_rl_abs, test_folder) if avg_rl_abs else ""

    template_dir = os.path.join(os.path.dirname(__file__), 'report_templates')
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

    # Generate HTML with image references
    template = env.get_template("report.html")
    total = calc_summary("total", stats)
    # TODO: provide correct units of measurements for use_segments=false, %, deg
    html = template.render(
        date=date_time,
        branch_name=branch_name,
        commit_sha=commit_sha,
        commit_ts=commit_ts,
        provenance_warning=provenance_warning,
        comments=comments,
        stats=stats_for_html,
        total=total,
        avg_tl_path=avg_tl_rel,
        avg_rl_path=avg_rl_rel,
    )

    html_file_name = os.path.join(test_folder, f"{report_basename}.html")
    with open(html_file_name, "wt") as f:
        f.write(html)

    print(f"{html_file_name} was done")

    # Generate PDF if requested (with embedded base64 images)
    if generate_pdf:
        try:
            from weasyprint import HTML

            # For PDF: convert images to base64 for embedding
            stats_for_pdf = []
            for s in stats:
                stat_dict = {
                    'sequence_title': s.sequence_title,
                    'n_frames': s.n_frames,
                    'average_fps': s.average_fps,
                    'gt_av_translation_error': s.gt_av_translation_error,
                    'gt_av_rotation_error': s.gt_av_rotation_error,
                    'gt_simple_error': s.gt_simple_error,
                    'bird_view_with_errors_path': s.bird_view_with_errors_path,
                    'bird_view_base64': image_to_base64(s.bird_view_with_errors_path) if s.bird_view_with_errors_path else ""
                }
                stats_for_pdf.append(stat_dict)

            # Use PDF-specific template
            pdf_template = env.get_template("report_pdf.html")
            pdf_html = pdf_template.render(
                date=date_time,
                branch_name=branch_name,
                commit_sha=commit_sha,
                commit_ts=commit_ts,
                provenance_warning=provenance_warning,
                comments=comments,
                stats=stats_for_pdf,
                total=total,
                avg_tl_base64=image_to_base64(avg_tl_abs) if avg_tl_abs else "",
                avg_rl_base64=image_to_base64(avg_rl_abs) if avg_rl_abs else "",
            )

            pdf_file_name = os.path.join(test_folder, f"{report_basename}.pdf")
            HTML(string=pdf_html).write_pdf(pdf_file_name)
            print(f"{pdf_file_name} was done")
        except ImportError as exc:
            raise RuntimeError(
                "PDF generation requires the optional PDF dependencies; install cuvslam-tools[pdf]."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"PDF generation failed: {exc}") from exc
