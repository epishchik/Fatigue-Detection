NAME_IMAGE=blinking
NAME_CONTEINER=blinking
PATH_TO_PRJ=/proj
WORKSPACE=/workspace/proj
GPUS='0'

NV_GPU=$GPUS nvidia-docker run \
    --rm \
    -it \
    --shm-size 32G \
    -v $PATH_TO_PRJ:$WORKSPACE \
    --name $NAME_CONTEINER \
    $NAME_IMAGE