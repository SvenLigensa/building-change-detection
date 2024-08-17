# Building-Change Detection
This is the repository accompanying the thesis *"Understanding Building-Change Detection"* which I wrote as part of the seminar *Advanced Topics in Machine Learning* (ST 2024).

In the theoretical part, some models for Change Detection (CD) introduced in the last years were presented and in the empirical part, they were benchmarked and their performance analyzed.
## Getting Started
This repository is based on the [open-cd](https://github.com/likyoo/open-cd "open-cd") toolbox. To obtain a working installation, I performed the following steps:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U openmim
pip install torch torchvision --no-cache-dir
pip install pytz==2023.3
pip install requests==2.28.2
pip install rich==13.4.2
pip install setuptools==60.2.0
pip install tqdm==4.65.0
pip install setuptools==60.2.0
pip install --upgrade pip setuptools wheel
pip install --force-reinstall six
pip install --force-reinstall setuptools
pip install --force-reinstall openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpretrain>=1.0.0rc7"
pip install "mmsegmentation>=1.2.2"
pip install "mmdet>=3.0.0"
git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .
pip install torch==2.0.0 torchvision==0.15.1 --no-cache-dir
mim install mmcv==2.0.0
pip install ftfy
pip install regex
```
## Model Benchmarking  
The configuration files and logs of model training and evaluation can be found in the `models` directory.

The trained model checkpoints are available [here](TODO SCIEBO LINK "Model Checkpoints"), and the results when applying them to the three datasets `S2Looking`, `LEVIR-CD` and `WHU-CD` are available [here](TODO SCIEBO LINK "Results").

All commands are executed at the root directory which is organized as follows:  

```
├── data/
│ ├── S2Looking/
│ ├── LEVIR-CD/
│ └── WHU/
├── open-cd/
├── fc_ef/
├── fc_siam_conc/
├── fc_siam_diff/
├── changestar/
├── changeformer/
├── changer/
├── ttp/
└── venv/
```

Where `data` is the directory containing the three datasets: `S2Looking`, `LEVIR-CD`, and `WHU`.
`venv` is the virtual environment to which the dependencies of the project are installed.
`open-cd` is the repository cloned from *https://github.com/likyoo/open-cd.git*, as mentioned above.
Further, there is one directory per model, which contains their logs, configurations, and checkpoints.
In this repo, they can be found in the `models` directory.

The following commands were executed for *training* and *evaluation* of the models.  
```
$ python open-cd/tools/train.py open-cd/configs/{opencd_model_dir}/{train_config}.py --work-dir ./{model_dir}
```

```
$ python open-cd/tools/test.py open-cd/configs/{opencd_model_dir}/{train_config}.py {model_dir}/best_checkpoint.pth --work-dir ./{model_dir} --show-dir ./{model_dir}/{s2looking / levircd / whu}
```

| Model        | opencd_model_dir | model_dir    | train_config                                 | test_config                                                                                                                          |
| ------------ | ---------------- | ------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| FC EF        | fcsn             | fc_ef        | fc_ef_512x512_s2looking                      | fc_ef_512x512_s2looking / fc_ef_512x512_eval_levircd / fc_ef_512x512_eval_whu                                                        |
| FC Siam Conc | fcsn             | fc_siam_conc | fc_siam_conc_512x512_s2looking               | fc_siam_conc_512x512_s2looking / fc_siam_conc_512x512_eval_levircd / fc_siam_conc_512x512_eval_whu                                   |
| FC Siam Diff | fcsn             | fc_siam_diff | fc_siam_diff_512x512_s2looking               | fc_siam_diff_512x512_s2looking / fc_siam_diff_512x512_eval_levircd / fc_siam_diff_512x512_eval_whu                                   |
| ChangeSTAR   | changestar       | changestar   | changestar_farseg_1x96_512x512_60k_s2looking | changestar_farseg_1x96_512x512_60k_s2looking / changestar_farseg_1x96_512x512_eval_levircd / changestar_farseg_1x96_512x512_eval_whu |
| ChangeFormer | changeformer     | changeformer | changeformer_mit-b0_512x512_1M_s2looking     | changeformer_mit-b0_512x512_1M_s2looking / changeformer_mit-b0_512x512_eval_levircd / changeformer_mit-b0_512x512_eval_whu           |
| Changer      | changer          | changer      | changer_512x512_60k_s2looking                | changer_512x512_60k_s2looking / changer_512x512_eval_levircd / changer_512x512_eval_whu                                              |
| TTP          | ttp              | ttp          | ttp_vit-sam-l_512x512_1M_s2looking           | ttp_vit-sam-l_512x512_1M_s2looking / ttp_vit-sam-l_512x512_eval_levircd / ttp_vit-sam-l_512x512_eval_whu                             |
## Datasets Download and Preprocessing
**LEVIR-CD** is small enough that it could be downloaded from GoogleDrive (https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim) and uploaded via the UI to the Jupyter server.
**S2Looking** was too big for that process, and `wget` could not be used with GoogleDrive due to a popup window. So I downloaded the dataset to my own machine, uploaded it on the filehosting service *Filebin*, and downloaded it to the server from there via `wget`.
The download of **WHU-CD** via `wget` took multiple hours due to the bad internet connection to a server presumably located in China. Afterwards, the files needed to be moved to other subdirectories for `open-cd` to be able to handle them:
1. As we want to use the whole dataset for evaluation, we want to move all files into one subdirectory. Because some of the file names are appearing twice, the prefix `test_` is added to alll images in subfolders of `/test` (analogously `train_` for images in `/train`)
2. Images in the `/splited_images` subfolder of `/2012` are moved to the folder `/A` (analogously `/2016` to `/B`)
3. `/train` and `/test` subfolders are merged in `/A` (and in `/B`)
4. Change detection labels were generated from two individual labels via `xor`ing them (the slight misalignment was ignored)
