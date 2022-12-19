# Start inference

# wide
#python3 inference.py --n_class 11 --dataset tree --checkpoint ./run/tree/resnet/model_best.pth.tar --output_directory ./run/tree/resnet/inference_output --image_path /media/qisens/4tb3/scribble_dataset/0103_not_aug_overfit/used_img/

# tight
#python3 inference.py --n_class 11 --dataset tree --checkpoint ./run/tree/resnet/model_best.pth.tar --output_directory ./run/tree/resnet/inference_output --image_path /media/qisens/4tb3/scribble_dataset/building_capture/flat_facility_round_edge_corrugated_slate_panel_garden/original_png/

# test img
python3 inference.py --n_class 12 --dataset tree --checkpoint ./run/tree/resnet/model_best.pth.tar --output_directory ./run/tree/resnet/inference_output --image_path /media/qisens/4tb3/scribble_dataset/scribble_test_area/original_png/
