pip install -r requirements.txt
# Uninstall existing PyTorch (if any)
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA support
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4

cd preannotation_pipeline/rf-detr/rfdetr/
python 01_miner.py --yolo_thresh 0.7 --rfdetr_thresh 0.5

python 02_packager.py
