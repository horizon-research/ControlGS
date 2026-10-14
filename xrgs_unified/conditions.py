from __future__ import annotations

import math
import random


def display_condition_alpha(saving_ratio: float, saving_max: float) -> float:
    saving_ratio = max(float(saving_ratio), 0.0)
    saving_max = max(float(saving_max), 1e-6)
    return min(saving_ratio / saving_max, 1.0)


def sample_downscale(args) -> float:
    low = max(float(args.downscale_min), 1.0)
    high = max(float(args.downscale_max), low)
    if high <= low:
        return low
    log_low = math.log(low, 2.0)
    log_high = math.log(high, 2.0)
    return 2.0 ** (log_low + (log_high - log_low) * random.random())


def sample_target_saving(args) -> float:
    low = float(args.target_saving_min)
    high = float(args.target_saving_max)
    if high <= low:
        return low
    return low + (high - low) * random.random()


def set_unified_conditions(camera, downscale: float, saving_ratio: float, args) -> None:
    ss_scale_max = max(float(args.ss_scale_max), 1.000001)
    camera.ss_res_cond = math.log(max(float(downscale), 1.0), 2.0) / math.log(ss_scale_max, 2.0)
    camera.res_cond = camera.ss_res_cond
    camera.display_res_cond = display_condition_alpha(saving_ratio, args.target_saving_max)
    camera.display_target_saving = float(saving_ratio)
    camera.lens_condition_radius_max_norm = args.lens_condition_radius_max_norm
