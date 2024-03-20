# EQLM

Extreme Q-Learning Machine (EQLM) is a Q-learning[1] algorithm which performs network weight updates using ELM[2]. Specifically, the upgrade mechanism is an incremental form of ELM referred to as LS-IELM[3].

## Citation

If you use this code in your work please cite the following paper available [here](https://arxiv.org/abs/2006.02986) 

"Wilson C, Riccardi A, Minisci E. A novel update mechanism for Q-Networks based on extreme learning machines. arXiv preprint arXiv: 2006.02986"

Bibtex entry:
```
@misc{wilson2020novel,
    title={A Novel Update Mechanism for Q-Networks Based On Extreme Learning Machines},
    author={Callum Wilson and Annalisa Riccardi and Edmondo Minisci},
    year={2020},
    eprint={2006.02986},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

## Environment
```
Platform: Window with GPU
```

### Install the reqiuremnt.txt
```
For Windows:
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 --yes
# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11" # with CPU and GPU
# Verify the installation:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"



conda install --yes matplotlib multiprocess pandas scipy sympy tqdm XlsxWriter pygraphviz jupyter chardet
pip install scikit_learn==1.2.2 bootstrapped

conda install -c conda-forge deap gym
```

## References
1. Watkins CJ, Dayan P. Q-learning. Machine learning. 1992 May 1;8(3-4):279-92.
2. Huang GB, Zhu QY, Siew CK. Extreme learning machine: theory and applications. Neurocomputing. 2006 Dec 1;70(1-3):489-501.
3. Guo L, Hao JH, Liu M. An incremental extreme learning machine for online sequential learning problems. Neurocomputing. 2014 Mar 27;128:50-8.
