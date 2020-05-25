export PYTHONUNBUFFERED=1
sbatch --export=ALL \
    -J metm \
    --time=30:00:00 \
    --account=ctb-liyue \
    --gres=gpu:1 \
    --output slurm_output/slurm-%j.out \
    -c 2 \
    --qos=high \
    --mem=30G \
    --mail-type=ALL \
    --mail-user=zhi.wen@mail.mcgill.ca \
    ./run.sh
