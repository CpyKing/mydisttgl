rm -r build/
python setup.py build_ext --inplace
python gen_minibatch.py --data WIKI
