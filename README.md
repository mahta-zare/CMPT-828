The contents of this repository were borrowed from https://github.com/motokimura/spacenet8_solution_5th-place and modified to focus on the road network extraction task. 

This code was run on the compute node Narval of Digitial Research Alliance of Canada.

Here are the steps to run this code:

Downloading the data from AWS s3 bucket:

To downlaod the data, one should have a free AWS account and have generated an aws access key.
Installing AWS CLI:

    module load python
    pip install awscli awscli-plugin-endpoint
    aws configure

Enter your aws access key and secret access key. Set the default region name to 'us-east-2' and 
the output format as 'json'.

Creating the directories:
To create the directories cd to the directory of your choise and do:
(I used the project space provided in Narval)

    mkdir -p /data/spacenet8/train
    mkdir -p /data/spacenet8/test
    mkdir -p /data/spacenet8_artifact
    chmod 777 -R data
    cd /data/spacenet8

When AWS CLI is installed and the directories are created we can download the data by entering the following commands:
    aws s3 cp --recursive s3://spacenet-dataset/spacenet/SN8_floods/tarballs/ .
    aws s3 cp s3://spacenet-dataset/spacenet/SN8_floods/resolutions.txt .

The following commands extract the tar files and put them in their spesific directories:

    mkdir train/Germany_Training_Public
    mv Germany_Training_Public.tar.gz train/Germany_Training_Public/
    cd train/Germany_Training_Public/
    tar -xvf Germany_Training_Public.tar.gz 
    rm Germany_Training_Public.tar.gz

    cd ../..
    mkdir train/Louisiana-East_Training_Public
    mv Louisiana-East_Training_Public.tar.gz train/Louisiana-East_Training_Public/
    cd train/Louisiana-East_Training_Public/
    tar -xvf Louisiana-East_Training_Public.tar.gz
    rm Louisiana-East_Training_Public.tar.gz

    cd ../..
    mkdir test/Louisiana-West_Test_Public
    mv Louisiana-West_Test_Public.tar.gz test/Louisiana-West_Test_Public/
    cd test/Louisiana-West_Test_Public/
    tar -xvf Louisiana-West_Test_Public.tar.gz
    rm Louisiana-West_Test_Public.tar.gz

    cd ../..
    cp resolutions.txt train/
    cp resolutions.txt test/
    rm resolutions.txt


Create a virtual environement:

load python:
    module load python scipy-stack
    module load gcc/9.3.0  cuda/11.4
    module load opencv/4.7.0
    module load gdal


Create a virtual environment named "deep" and install the nessecary packages:
    virtualenv --no-download deep
    source deep/bin/activate
    pip install --no-index --upgrade pip
    pip install --no-index -r requirements.txt

where requirements.txt includes all the necessary packages that should be installed.

We also need to build libgeos/geos from the source from this git repository: 
https://github.com/libgeos/geos

We are now ready to run our code:

    python tools/make_folds_v3.py --train_dir $TRAIN_DIR
    python tools/prepare_road_masks.py --train_dir $TRAIN_DIR


    mkdir -p /wdata/mosaics/
    python tools/mosaic.py --fold_id 0 --train_dir $TRAIN_DIR > /wdata/mosaics/train_0.txt 2>&1 &
    python tools/mosaic.py --fold_id 1 --train_dir $TRAIN_DIR > /wdata/mosaics/train_1.txt 2>&1 &
    python tools/mosaic.py --fold_id 2 --train_dir $TRAIN_DIR > /wdata/mosaics/train_2.txt 2>&1 &
    python tools/mosaic.py --fold_id 3 --train_dir $TRAIN_DIR > /wdata/mosaics/train_3.txt 2>&1 &
    python tools/mosaic.py --fold_id 4 --train_dir $TRAIN_DIR > /wdata/mosaics/train_4.txt 2>&1 &

training:


    mkdir /work/xdxd_sn5_models
    wget -nv https://motokimura-public-sn8.s3.amazonaws.com/xdxd_sn5_serx50_focal.zip
    unzip xdxd_sn5_serx50_focal.zip && rm -f xdxd_sn5_serx50_focal.zip

    export LOG_DIR=/wdata/logs/train
    mkdir -p $LOG_DIR

    export ARGS=" --override_model_dir /work/models --disable_wandb Data.train_dir=$TRAIN_DIR"

    export LOG_DIR=/wdata/logs/train
    mkdir -p $LOG_DIR

    python tools/train_net.py \
        --task foundation_xdxd_sn5 \
        --exp_id 80000 \
        --fold_id 0 \
        --disable_wandb
        --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold0/fold0_best.pth \
        $ARGS \
        > $LOG_DIR/exp_80000.txt 2>&1 &

    python tools/train_net.py \
        --task foundation_xdxd_sn5 \
        --exp_id 80001 \
        --fold_id 1 \
        --disable_wandb
        --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold1/fold1_best.pth \
        $ARGS \
        > $LOG_DIR/exp_80001.txt 2>&1 &

    python tools/train_net.py \
        --task foundation_xdxd_sn5 \
        --exp_id 80002 \
        --fold_id 2 \
        --disable_wandb
        --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold2/fold2_best.pth \
        $ARGS \
        > $LOG_DIR/exp_80002.txt 2>&1 &

    python tools/train_net.py \
        --task foundation_xdxd_sn5 \
        --exp_id 80003 \
        --fold_id 3 \
        --disable_wandb
        --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold3/fold3_best.pth \
        $ARGS \
        > $LOG_DIR/exp_80003.txt 2>&1 &


    python tools/train_net.py \
        --task foundation_xdxd_sn5 \
        --exp_id 80004 \
        --fold_id 4 \
        --disable_wandb
        --pretrained_path /work/xdxd_sn5_models/xdxd_sn5_serx50_focal/fold0/fold0_best.pth \
        $ARGS \
        > $LOG_DIR/exp_80004.txt 2>&1 &


testing:

    export LOG_DIR=/wdata/logs/test
    mkdir -p $LOG_DIR

    python tools/test_net.py --exp_id 80000 --artifact_dir /wdata --override_model_dir /work/models Data.test_dir=/data/test > $LOG_DIR/exp_80000.txt 2>&1 &
    python tools/test_net.py --exp_id 80001 --artifact_dir /wdata --override_model_dir /work/models Data.test_dir=/data/test > $LOG_DIR/exp_80001.txt 2>&1 &
    python tools/test_net.py --exp_id 80002 --artifact_dir /wdata --override_model_dir /work/models Data.test_dir=/data/test > $LOG_DIR/exp_80002.txt 2>&1 &
    python tools/test_net.py --exp_id 80003 --artifact_dir /wdata --override_model_dir /work/models Data.test_dir=/data/test > $LOG_DIR/exp_80003.txt 2>&1 &
    python tools/test_net.py --exp_id 80004 --artifact_dir /wdata --override_model_dir /work/models Data.test_dir=/data/test > $LOG_DIR/exp_80004.txt 2>&1 &

    python tools/ensemble.py --exp_id 80000 80001 80002 80003 80004 --root_dir /data/test --artifact_dir /wdata
