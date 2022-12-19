 # Resume training
#python3 tree_train_withdensecrfloss.py --backbone resnet --lr 0.0007 --workers 1 --epochs 10000 --batch-size 8  --checkname resnet --eval-interval 1 --dataset tree --save-interval 100 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --resume ./run/tree/resnet/experiment_0/checkpoint_epoch_1200.pth.tar

# Start training
python3 tree_train_withdensecrfloss.py --backbone resnet --lr 0.0007 --workers 1 --epochs 100000 --batch-size 8 --checkname resnet --eval-interval 1 --dataset tree --save-interval 100 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100



