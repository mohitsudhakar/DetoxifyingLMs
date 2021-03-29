conda activate mohit

nohup python code/detoxbert.py > bert.log &
nohup python code/detoxgpt2.py > gpt2.log &
nohup python code/detoxroberta.py > roberta.log &
nohup python code/detoxxlnet.py > xlnet.log &

