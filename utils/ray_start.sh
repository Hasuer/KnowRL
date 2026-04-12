#!/bin/bash

export HYDRA_FULL_ERROR=1


MASTER_ADDR=$(echo $PADDLE_TRAINERS | cut -d',' -f1)
MASTER_PORT=6379
DASHBOARD_PORT=8265
RANK=$PADDLE_TRAINER_ID
WORLD_SIZE=$PADDLE_TRAINERS_NUM
LOCAL_IP=$TRAININGJOB_SERVICE


if [[ "$1" == "count" ]]; then
    output=$(ray status)
    active_node_count=$(echo "$output" | grep -o "node_[a-f0-9]\+" | wc -l)
    echo "Active nodes: $active_node_count"
    exit 0
fi


if [[ "$RANK" -eq "0" ]]; then
    echo "Main Node: RANK=$RANK"
    rm -rf /tmp/
    rm -rf /tmp/
    rm -rf /tmp/
    ray start --head --dashboard-host='0.0.0.0' --port=${MASTER_PORT} --dashboard-port=${DASHBOARD_PORT}
    status=$?

    if [ $status -ne 0 ]; then
        echo "Ray failed to start, exit code: $status"
        exit $status
    fi

# Worker node logic
elif [[ "$RANK" -gt "0" && "$RANK" -lt "16" ]]; then
    echo "Worker Node: RANK=$RANK"

    while :
    do
        rm -rf /tmp/
        rm -rf /tmp/
        rm -rf /tmp/

        if ray start --address=${MASTER_ADDR}:${MASTER_PORT} --node-ip-address="${LOCAL_IP}"; then
            echo "Ray worker started successfully."
            break
        else
            echo "Ray worker failed to start, retrying in 3 seconds..."
            sleep 3
        fi
    done

    echo "Worker node is running..."
else
    echo "RANK=$RANK is out of valid range (0 ~ 15)"
    exit 1
fi
