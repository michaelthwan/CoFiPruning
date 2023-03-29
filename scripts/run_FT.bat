@echo off

:: Example run: run_FT.bat [TASK] [EX_NAME_SUFFIX]

set glue_low=MRPC RTE STS-B CoLA
set glue_high=MNLI QQP QNLI SST-2

set proj_dir=%n%\space2

set code_dir=%proj_dir%\CoFiPruning

:: task and data
set task_name=%1
set data_dir=%proj_dir%\data\glue_data\%task_name%

:: pretrain model
set model_name_or_path=bert-base-uncased

:: logging & saving
set logging_steps=100
set save_steps=0

echo %glue_low% | findstr /C:%task_name% >nul 2>&1 && set eval_steps=50
echo %glue_high% | findstr /C:%task_name% >nul 2>&1 && set eval_steps=500

:: train parameters
set max_seq_length=128
set batch_size=32
set learning_rate=2e-5
set epochs=5

:: seed
set seed=57

:: output directory
set ex_name_suffix=%2
set ex_name=%task_name%_%ex_name_suffix%
set output_dir=%proj_dir%\out-test\%task_name%\%ex_name%
mkdir %output_dir% >nul 2>&1
set pruning_type=None

python %code_dir%\run_glue_prune.py ^
	   --output_dir %output_dir% ^
	   --logging_steps %logging_steps% ^
	   --task_name %task_name% ^
	   --data_dir %data_dir% ^
	   --model_name_or_path %model_name_or_path% ^
	   --ex_name %ex_name% ^
	   --do_train ^
	   --do_eval ^
	   --max_seq_length %max_seq_length% ^
	   --per_device_train_batch_size %batch_size% ^
	   --per_device_eval_batch_size 32 ^
	   --learning_rate %learning_rate% ^
	   --num_train_epochs %epochs% ^
	   --overwrite_output_dir ^
	   --save_steps %save_steps% ^
	   --eval_steps %eval_steps% ^
	   --evaluation_strategy steps ^
	   --seed %seed% 2>&1 | tee %output_dir%\all_log.txt
