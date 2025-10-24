python /data5/laiping/tianzhibei/code/object_detection/test_one_stage_recognition.py \
 --gallery_path /data5/laiping/tianzhibei/outputs/20250721/input \
 --xml_output_path /data5/laiping/tianzhibei/outputs/20250721/xml_output \
 --eval-options submission_dir=/data1/zhuhongchun/outputs/20250721/work_dirs \
 --checkpoint /data5/laiping/tianzhibei/checkpoint/lsknet_s_fair_epoch12.pth \
 --config /data5/laiping/tianzhibei/code/object_detection/configs/lsknet/lsk_s_fpn_1x_fair_le90.py \
 --save_dirs /data5/laiping/tianzhibei/outputs/20250721/test \
 --evaluation False