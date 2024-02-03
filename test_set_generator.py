# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import os
import random
import glob

test_json_index = [9308, 6789, 516, 6590, 9297, 6852, 766, 2714, 7296, 1046, 4247, 2583, 7313, 8643, 7983, 9198, 1, 637,
                   8103, 5340, 5112, 7649, 816, 6801, 3080, 8987, 1367, 2138, 241, 6583, 6840, 5180, 55, 3498, 234, 38,
                   8656, 1602, 3120, 1948, 3252, 4954, 4587, 2985, 1641, 7792, 6499, 1332, 357, 4500, 7421, 1896, 4202,
                   2185, 8533, 5686, 1885, 2530, 4561, 304, 693, 666, 3370, 4254, 9148, 5156, 6011, 688, 8101, 7514,
                   7134, 6102, 8813, 2921, 3405, 6153, 4768, 145, 2268, 2474, 4446, 5462, 5529, 6016, 1535, 5541, 584,
                   675, 4417, 2684, 2448, 4743, 5913, 6468, 8986, 2124, 4807, 1882, 7832, 3927, 790, 5044, 2942, 8569,
                   1161, 4958, 6605, 5382, 4902, 6794, 1780, 1628, 9188, 7883, 7766, 5522, 5630, 2036, 7849, 1900, 8154,
                   6988, 619, 4946, 5489, 2550, 2728, 9248, 1424, 1078, 1387, 3244, 3621, 1001, 6304, 128, 1606, 6452,
                   9117, 8503, 4748, 7348, 8005, 3559, 6931, 1370, 6034, 3606, 4274, 2729, 7065, 3144, 5873, 1884, 452,
                   8614, 7399, 3304, 8144, 6520, 3395, 689, 3537, 2397, 1714, 3243, 7510, 6193, 5925, 8952, 2479, 1716,
                   7993, 2431, 9239, 6650, 6934, 8539, 8116, 5283, 8166, 8169, 3309, 8893, 3584, 159, 5574, 5214, 5272,
                   581, 8603, 2430, 4208, 2554, 6209, 4823, 7707, 1087, 1386, 8462]


def prepare_test_and_validation_data_files(emulated_dataset_dir, test_dataset_dir, validation_set_dir):
    # prepare test data
    for num in test_json_index:
        source_files_dir = os.path.join(emulated_dataset_dir, str(num).zfill(5) + '.json')
        target_files_dir = os.path.join(test_dataset_dir, str(num).zfill(5) + '.json')
        shutil.move(source_files_dir, target_files_dir)

    shutil.move(os.path.join(emulated_dataset_dir, "baseline_eval.pkl"), os.path.join(test_dataset_dir, "baseline_eval.pkl"))

    # prepare validation data
    train_data_files = glob.glob(os.path.join(emulated_dataset_dir, f'*.json'), recursive=True)
    train_data_files = random.sample(train_data_files, 200)
    for files in train_data_files:
        target_files_dir = os.path.join(validation_set_dir, os.path.basename(files))
        shutil.copy(files, target_files_dir)


if __name__ == "__main__":
    random.seed(666)
    emulated_dataset_dir = "../../train_data/emulated_dataset"
    validation_set_dir = "../../validation_set"
    test_dataset_dir = "../../test_set"

    if not os.path.exists(validation_set_dir):
        os.mkdir(validation_set_dir)

    if not os.path.exists(test_dataset_dir):
        os.mkdir(test_dataset_dir)

    prepare_test_and_validation_data_files(emulated_dataset_dir, test_dataset_dir, validation_set_dir)
