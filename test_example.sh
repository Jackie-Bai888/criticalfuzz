: '
To use the scripts, the output (i.e., -o) of DeepHunter should follow the structure: root_dir/strategy/metrics/id
The strategy and metrics must be the same name with the option, i.e., strategy must be one of [random,uniform,tensorfuzz,deeptest,prob]
metrics must be one of [nbc,snac,tknc,kmnc,nc]. To get the coverage of random strategy in terms of a specific metric, we also need to select the specific metric.
id can be any number.

Before using the new scripts, please install the xxhash by "pip install xxhash"
'
python image_fuzzer.py  -i ../test_seeds/mnist_seeds  -o lenet1_out/prob/nc/0 -model lenet1 -criteria nc -random 0 -select prob -max_iteration 200
python image_fuzzer.py  -i ../test_seeds/mnist_seeds  -o lenet4_out/prob/nc/0 -model lenet4 -criteria nc -random 0 -select prob -max_iteration 200
python image_fuzzer.py  -i ../test_seeds/mnist_seeds  -o lenet5_out/prob/nc/0 -model lenet5 -criteria nc -random 0 -select prob -max_iteration 200
python image_fuzzer.py  -i ../test_seeds/cifar_seeds  -o resnet20_out/prob/nc/0 -model resnet20 -criteria nc -random 0 -select prob -max_iteration 200

python utils/CoveragePlot.py -i lenet5_out -type coverage -iterations 200 -o  results/coverage_plot.pdf
python utils/CoveragePlot.py -i lenet5_out -type seedattack -iterations 200 -o  results/diverse_plot.pdf
python utils/UniqCrashBar.py -i lenet5_out -iterations 200 -o  results/uniq_crash.pdf
echo 'Finish! Please find the results in the results directory.'

