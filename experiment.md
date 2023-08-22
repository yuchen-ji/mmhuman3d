## Train

```sh
# kill distribution training
kill $(ps aux | grep tools/train.py | grep -v grep | awk '{print $2}')
# train model
nohup bash tools/dist_train.sh configs/ormr/hrnet_w32_ormr_w_htmp_crop_adv.py workspace/ormr/epoch10_w_htmp_crop_adv 4 > workspace/ormr/nohup.log 2>&1 &
# eval model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash tools/dist_test.sh workspace/ormr/epoch8_w_htmp_crop_adv/hrnet_w32_ormr_w_htmp_crop_adv.py workspace/ormr/epoch8_w_htmp_crop_adv workspace/ormr/epoch8_w_htmp_crop_adv/epoch_3.pth 6 --metrics=mpjpe 
```