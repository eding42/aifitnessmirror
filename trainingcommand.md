#replace  with your own data path

python Edward-train-DNN.py   --data-dir /home/ed/aifitnessmirror/data/videos   --yolo yolov8n-pose.pt   --imgsz 256   --simulate-camera 256   --sample-rate 1   --window 16 --stride 4   --epochs 30 --batch-size 256   --out-dir /home/ed/
aifitnessmirror/output/exercise_model   --name EXERCISE