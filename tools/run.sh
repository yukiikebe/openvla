xhost +local:root
export WANDB_API_KEY=60c883aff30a57af75e35c552172dbd07a2c9a2c

docker run --shm-size=16g -it --entrypoint /bin/bash -v /tmp/.X11-unix:/tmp/.X11-unix \
    -p 8000:8000 \
    -v /home/yuki/research/openvla:/workspace \
    -v /mnt/crucial/yuki/:/mnt/crucial/yuki \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e DISPLAY=$DISPLAY --gpus all yuki/openvla:latest