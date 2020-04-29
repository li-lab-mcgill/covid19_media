export PYTHONUNBUFFERED=1
sbatch --export=ALL --account=ctb-liyue --gres=gpu:1 -c 1 --qos=high --mem=30G ./run.sh