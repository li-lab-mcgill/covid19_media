export PYTHONUNBUFFERED=1
sbatch --export=ALL \
    -J metm \
    --account=ctb-liyue \
    --gres=gpu:1 \
    -c 1 \
    --qos=high \
    --mem=30G \
    --mail-type=ALL \
    --mail-user=zhi.wen@mail.mcgill.ca \
    ./run.sh