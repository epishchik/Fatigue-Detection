NAME_IMAGE=blinking
NAME_CONTEINER=blinking
PATH_TO_PRJ=/proj
WORKSPACE=/workspace/proj
GPUS='9'

NV_GPU=$GPUS nvidia-docker run \
    --rm \
    -it \
    --shm-size 64G \
    -v $PATH_TO_PRJ:$WORKSPACE \
    --name $NAME_CONTEINER \
    $NAME_IMAGE