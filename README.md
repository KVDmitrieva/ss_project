# Source Separation project

## Installation guide

```shell
pip install -r ./requirements.txt
```
## Train running guide
Add your model config to `hw_ss/configs/` and run:

```shell
python3 train.py -c hw_ss/configs/your_config.json
```
In order to recreate results, use `train_jasper.json`:
```shell
python3 train.py -c hw_ss/configs/spex.json
```
By default, config assumes that it is used in kaggle with [librispeechmix](https://www.kaggle.com/datasets/katedmitrieva/librispeechmix) dataset. 
It provides ~10k training and 500 validation triplets. If you want to generate new triplets you need to remove 
`mixture_dir` from config and set `generate_mixture` flag to `true`, also provide correct librispeech `part`.
## Test running guide
First of all, download model checkpoint and its config:
```shell
cd default_test_model
wget "https://www.dropbox.com/scl/fi/nk22f0b0aetaoqh687krp/model_best-3.pth?rlkey=uov16d29r24ldq1fvky4tsi0n&dl=0" -O model.pth
wget "https://www.dropbox.com/scl/fi/93lig6rfbofcfi11qdmf1/config.json.1?rlkey=t6ythihizlv1e5y9gan8mdlg1&dl=0" -O config.json
cd ..
```
Run test-clean
```shell
python3 test.py \
   -c default_test_model/config.json \
   -r default_test_model/model.pth \
   -o test_result.json \
   -t DIR_PATH
```
`DIR_PATH` should contain three dirs: mix (for mixes), refs (for references) and targets (for targets). 
Files in each dir are named in the following way ID-mixed.wav, ID-ref.wav, ID-target.wav for mix, ref and target respectively.

After running test, `test_result.json` file should be created. All metrics would be written at the end of the file.

## ASR Bonus
After downloading checkpoints run:
```shell
python3 test_asr.py \
   -c default_test_model/test_spex.json \
   -r default_test_model/model.pth
```
this would create `output` dir with `prediction` and `target` subdirs. Both contain test data for asr model. 

Follow the installation guide for [asr](https://github.com/KVDmitrieva/asr_project), download checkpoints and then run
```shell
python3 test.py \
   -c default_test_model/test_lm_jasper_clean.json \
   -r default_test_model/checkpoint.pth \
   -o test_result_clean.json \
   -t "YOUR_WORKIN_DIR/ss_project/output/prediction"
```
and
```shell
python3 test.py \
   -c default_test_model/test_lm_jasper_clean.json \
   -r default_test_model/checkpoint.pth \
   -o test_result_clean.json \
   -t "YOUR_WORKIN_DIR/ss_project/output/target"
```
in order to get result metrics for ss_model audio output and target audio files.
