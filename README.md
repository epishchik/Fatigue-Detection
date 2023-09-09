# Fatigue-Detection
Fatigue detection using blinking frequency and duration.

## Pull Image
```bash
docker pull pe4eniks/fatigue-detection
```
OR
```bash
docker pull ghcr.io/pe4eniks/fatigue-detection:latest
```


## Clone source code
```bash
git clone https://github.com/Pe4enIks/Fatigue-Detection.git
PATH_TO_PROJ=./Fatigue-Detection
```

## Run container from Image
```bash
NAME_IMAGE=pe4eniks/fatigue-detection:latest
NAME_CONTAINER=blinking
PATH_TO_PRJ=./Fatigue-Detection

docker run \
    --rm \
    -it \
    --shm-size 32G \
    -v $PATH_TO_PRJ:/workspace/proj \
    --name $NAME_CONTAINER \
    $NAME_IMAGE
```

- NAME_IMAGE - can be the name of the locally created image, the id of the pulled image or the name of the pulled image.
- NAME_CONTAINER - name of the container that will be run from the image.
- PATH_TO_PRJ - path to the cloned repository.

## Fatigue inference
Currently supported for algorithm variant only.

### Step 1
Configure config_algo_fatigue.yaml.
```yaml
model_name: mp_face - keypoint model [mp_face | spiga].
retinaface_backbone: mobile0.25 - backbone model [mobile0.25 | resnet50].
use_cpu: yes - cpu usage option [no | yes].
prefix: ./video/ - path to folder with video.
input_video: blinking.mp4 - video name with extension.
fps: 60 - video fps.
frequency_fatigue_threshold: 0.501 - fatigue threshold for frequency (if val <= threshold -> fatigue detected).
aecd_fatigue_threshold: 1.059 - fatigue threshold for aecd (if val >= threshold -> fatigue detected).
```

### Step 2
Run inference script.
```bash
python inference_algo_fatigue.py
```

## Method inference
### Algorithm
#### Step 1
Configure config_algo.yaml.
```yaml
model_name: mp_face - keypoint model [mp_face | spiga].
retinaface_backbone: mobile0.25 - backbone model [mobile0.25 | resnet50].
use_cpu: yes - cpu usage option [no | yes].
prefix: ./video/ - path to folder with video.
input_video: blinking.mp4 - video name with extension.
fps: 60 - video fps.
output_video: algo_blinking.mp4 - output video name.
graphics: algo_blinking.png - output graphics name.
```

#### Step 2
Run inference script.
```bash
python inference_algo.py
```

### 2D CNN
#### Step 1
Train a 2D CNN model. You need to download the mEBAL dataset and set correct paths in each .py file.
```bash
python preprocess_mEBAL.py
python train.py
python test.py
```

#### Step 2
Configure config_cnn.yaml.
```yaml
model_name: mp_face - keypoint model [mp_face | spiga].
retinaface_backbone: mobile0.25 - backbone model [mobile0.25 | resnet50].
use_cpu: yes - cpu usage option [no | yes].
prefix: ./video/ - path to folder with video.
input_video: blinking.mp4 - video name with extension.
fps: 60 - video fps.
output_video: cnn_blinking.mp4 - output video name.
graphics: cnn_blinking.png - output graphics name.
```

#### Step 3
Run inference script.
```bash
python inference_cnn.py
```
