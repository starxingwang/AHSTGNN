# @Time     : 01. 07, 2022 16:57:
# @Author   : Xing Wang, Kexin Yang
# @FileName : generate_training_data.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/starxingwang/AHSTGNN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour):

    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour=12)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/Milan", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/data_mi_min.npy", help="Raw traffic readings.",)
    parser.add_argument("--num_for_predict", type=int, default=6, help="Sequence Length.",)
    parser.add_argument("--points_per_hour", type=int, default=6, help="Sequence Length.",)
    parser.add_argument("--num_of_weeks", type=int, default=1, help="", )
    parser.add_argument("--num_of_days", type=int, default=1, help="", )
    parser.add_argument("--num_of_hours", type=int, default=1, help="", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)

    df = np.load(args.traffic_df_filename).transpose(1, 0)
    num_samples, num_nodes = df.shape
    num_train = round(num_samples * 0.66)
    num_test = round(num_samples * 0.34)
    # num_train = 2880
    # num_test = 1440
    # num_val = num_samples - num_test - num_train
    train = df[:num_train]
    # val = df[num_train: num_train + num_val]
    test = df[-num_test:]

    hour_samples_train = []
    day_samples_train = []
    week_samples_train = []
    targets_train = []
    for idx in range(train.shape[0]):
        sample = get_sample_indices(train, args.num_of_weeks, args.num_of_days,
                                    args.num_of_hours, idx, args.num_for_predict,
                                    args.points_per_hour)
        if sample[-1] is None:
            continue

        week_sample, day_sample, hour_sample, target = sample

        hour_samples_train.append(hour_sample)
        day_samples_train.append(day_sample)
        week_samples_train.append(week_sample)
        targets_train.append(target)

    if hour_samples_train:
        hour_samples_train = np.stack(hour_samples_train, axis=0)
    if day_samples_train:
        day_samples_train = np.stack(day_samples_train, axis=0)
    if week_samples_train:
        week_samples_train = np.stack(week_samples_train, axis=0)
    targets_train = np.stack(targets_train, axis=0)

    hour_samples_train = np.expand_dims(hour_samples_train, 3)
    day_samples_train = np.expand_dims(day_samples_train, 3)
    week_samples_train = np.expand_dims(week_samples_train, 3)
    targets_train = np.expand_dims(targets_train, 3)

    print('hour_samples_train', hour_samples_train.shape)
    print('day_samples_train', day_samples_train.shape)
    print('week_samples_train', week_samples_train.shape)
    print('targets_train', targets_train.shape)

    # The ratios of training, validation and testing set are 2:0:1 for milan dataset,
    # so we commented out the part that generates the validation set
    # hour_samples_val = []
    # day_samples_val = []
    # week_samples_val = []
    # targets_val = []
    # for idx in range(val.shape[0]):
    #     sample = get_sample_indices(val, args.num_of_weeks, args.num_of_days,
    #                                 args.num_of_hours, idx, args.num_for_predict,
    #                                 args.points_per_hour)
    #     if sample[0] is None:
    #         continue
    #
    #     week_sample, day_sample, hour_sample, target = sample
    #
    #     hour_samples_val.append(hour_sample)
    #     day_samples_val.append(day_sample)
    #     week_samples_val.append(week_sample)
    #     targets_val.append(target)
    #
    # hour_samples_val = np.stack(hour_samples_val, axis=0)
    # day_samples_val = np.stack(day_samples_val, axis=0)
    # week_samples_val = np.stack(week_samples_val, axis=0)
    # targets_val = np.stack(targets_val, axis=0)
    #
    # hour_samples_val = np.expand_dims(hour_samples_val, 3)
    # day_samples_val = np.expand_dims(day_samples_val, 3)
    # week_samples_val = np.expand_dims(week_samples_val, 3)
    # targets_val = np.expand_dims(targets_val, 3)

    # print('hour_samples_val', hour_samples_val.shape)
    # print('day_samples_val', day_samples_val.shape)
    # print('week_samples_val', week_samples_val.shape)
    # print('targets_val', targets_val.shape)

    hour_samples_test = []
    day_samples_test = []
    week_samples_test = []
    targets_test = []
    for idx in range(test.shape[0]):
        sample = get_sample_indices(test, args.num_of_weeks, args.num_of_days,
                                    args.num_of_hours, idx, args.num_for_predict,
                                    args.points_per_hour)
        if sample[0] is None:
            continue

        week_sample, day_sample, hour_sample, target = sample

        hour_samples_test.append(hour_sample)
        day_samples_test.append(day_sample)
        week_samples_test.append(week_sample)
        targets_test.append(target)

    hour_samples_test = np.stack(hour_samples_test, axis=0)
    day_samples_test = np.stack(day_samples_test, axis=0)
    week_samples_test = np.stack(week_samples_test, axis=0)
    targets_test = np.stack(targets_test, axis=0)

    hour_samples_test = np.expand_dims(hour_samples_test, 3)
    day_samples_test = np.expand_dims(day_samples_test, 3)
    week_samples_test = np.expand_dims(week_samples_test, 3)
    targets_test = np.expand_dims(targets_test, 3)

    print('hour_samples_test', hour_samples_test.shape)
    print('day_samples_test', day_samples_test.shape)
    print('week_samples_test', week_samples_test.shape)
    print('targets_test', targets_test.shape)

    # for cat in ["train", "val", "test"]:
    #     _hour, _day, _week, _target = locals()["hour_samples_" + cat], locals()["day_samples_" + cat], locals()[
    #         "week_samples_" + cat], locals()["targets_" + cat]
    #     print(cat, "recent: ", _hour.shape, "day: ", _day.shape, "week: ", _week.shape, "target: ", _target.shape)
    #     np.savez_compressed(
    #         os.path.join('data/milan/data_period_12/', f"{cat}.npz"),
    #         hour=_hour,
    #         day=_day,
    #         week=_week,
    #         target=_target,
    #     )

    for cat in ["train", "test"]:
        _hour, _day, _week, _target = locals()["hour_samples_" + cat], locals()["day_samples_" + cat], locals()[
            "week_samples_" + cat], locals()["targets_" + cat]
        print(cat, "recent: ", _hour.shape, "day: ", _day.shape, "week: ", _week.shape, "target: ", _target.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            hour=_hour,
            day=_day,
            week=_week,
            target=_target,
        )