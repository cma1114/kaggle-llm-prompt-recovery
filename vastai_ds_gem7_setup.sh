# https://github.com/huggingface/alignment-handbook/blob/main/scripts/README.md
#ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes=2 ./lora_train_gem7_ds.py 

#ssh-keygen -t rsa -f vastai_key 
#ssh-add; ssh-add -l 
#mv ./vastai_key* ./.ssh
#ssh -i ./.ssh/vastai_key -p 40456 root@24.122.214.184
#scp -i ./.ssh/vastai_key -P 40456 -r /Users/christopherackerman/Downloads/mixed_dataset_big_train.csv root@24.122.214.184:/root/data/

echo 'export HF_TOKEN="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"' >> ~/.bashrc
echo 'export WANDB_API_KEY="5892f3b03c0c276c95ac7fca7a645ba07a53d953"' >> ~/.bashrc
source ~/.bashrc
chmod 600 ~/.bashrc

#Prep for gemma
pip3 install -q -U bitsandbytes==0.42.0
pip3 install -q -U peft==0.8.2
pip3 install -q -U trl==0.7.10
pip3 install -q -U accelerate==0.27.1
pip3 install -q -U datasets==2.17.0
pip3 install -q -U transformers==4.38.1
pip3 install wandb==0.15.11

pip3 install deepspeed
deepspeed --install_deps
