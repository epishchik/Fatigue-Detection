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
