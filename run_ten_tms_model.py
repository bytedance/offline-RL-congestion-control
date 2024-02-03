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

import glob
import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_dir = "./data"
    figs_dir = "./figs"
    onnx_model = "./onnx_model/TEN-TMS_model.onnx"
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)
    ort_session = ort.InferenceSession(onnx_model)

    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        baseline_model_predictions = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        for t in range(observations.shape[0]):
            feed_dict = {'obs': observations[t:t+1, :].reshape(1, 1, -1),
                         'hidden_states': hidden_state,
                         'cell_states': cell_state
                         }
            bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
            baseline_model_predictions.append(bw_prediction[0,0,0])
        baseline_model_predictions = np.asarray(baseline_model_predictions, dtype=np.float32)
        fig = plt.figure(figsize=(8, 5))
        time_s = np.arange(0, observations.shape[0]*60,60)/1000
        plt.plot(time_s, baseline_model_predictions/1000, label='ten_tms', color='g')
        plt.plot(time_s, bandwidth_predictions/1000, label='BW Estimator '+call_data['policy_id'], color='r')
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='k')
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Call Duration (second)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(figs_dir,os.path.basename(filename).replace(".json", ".png")))
        plt.close()