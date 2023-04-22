## Train

```sh
# kill distribution training
kill $(ps aux | grep tools/train.py | grep -v grep | awk '{print $2}')
# train model
nohup bash tools/dist_train.sh configs/ormr/hrnet_w32_ormr_w_htmp_wo_crop_adv.py workspace/ormr/test 4 > workspace/ormr/nohup.log 2>&1 &
```