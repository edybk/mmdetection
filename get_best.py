import json
acc = []
cascadercnn_logfile = "/data/home/bedward/workspace/mmpose-project/mmdetection/work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_coco_apas_allclass/20220716_162752.log.json"
yolox_logfile = "/data/home/bedward/workspace/mmpose-project/mmdetection/work_dirs/yolox_s_8x8_300e_apas_allclass/20220307_194236.log.json"

with open(yolox_logfile) as f:
    for l in f.readlines():
        d = json.loads(l)
        if "mode" in d and d['mode']=='val':
            acc.append((d.get("bbox_mAP", 0), d.get("epoch", "")))
print(max(acc))