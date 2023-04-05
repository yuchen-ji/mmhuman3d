CONFIG=$1
WORK_DIR=$2
CHECKPOINT=$3
GPUS=$4
PORT=${PORT:-29510}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py --config=${CONFIG} --work-dir=${WORK_DIR} --checkpoint=${CHECKPOINT} --launcher pytorch ${@:5}
