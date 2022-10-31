conda env create -f environment.yml
conda activate quant
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install gym
pip install tqsdk
pip install tqsdk2
pip install pymongo
pip install -U "ray[air]" # installs Ray + dependencies for Ray AI Runtime
pip install -U "ray[tune]"  # installs Ray + dependencies for Ray Tune
pip install -U "ray[rllib]"  # installs Ray + dependencies for Ray RLlib
pip install -U "ray[serve]"  # installs Ray + dependencies for Ray Serve
pip install autokeras
pip install -U modin
pip install pytorch-forecasting
