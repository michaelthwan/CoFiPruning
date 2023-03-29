@echo off
set "glue_low=(MRPC RTE STSB CoLA)"
set "glue_high=(MNLI QQP QNLI SST2)"
set "proj_dir=."
set "code_dir=%proj_dir%"

rem task and data
set "task_name=%1"
set "data_dir=%proj_dir%\data\glue_data\%task_name%"

rem pretrain model
set "model_name_or_path=bert-base-uncased"

rem logging & saving
set "logging_steps=100"
set "save_steps=0"

rem train parameters
set "max_seq_length=128"
set "batch_size=32"
set "learning_rate=2e-5"
set "reg_learning_rate=0.01"
set "epochs=20"

rem seed
set "seed=57"

rem output dir
set "ex_name_suffix=%2"
set "ex_name=%task_name%_%ex_name_suffix%"
set "ex_cate=%3"
set "output_dir=%proj_dir%\out\%task_name%\%ex_cate%\%ex_name%"

rem pruning and distillation
set "pruning_type=%4"
set "target_sparsity=%5"
set "distillation_path=%6"
set "distill_layer_loss_alpha=%7"
set "distill_ce_loss_alpha=%8"
set "distill_temp=2"
set "layer_distill_version=%9"

set "scheduler_type=linear"

rem if " %glue_low% " == " %task_name% " (
rem     set "eval_steps=50"
rem     set "epochs=100"
rem     set "start_saving_best_epochs=50"
rem     set "prepruning_finetune_epochs=4"
rem     set "lagrangian_warmup_epochs=20"
rem )
rem
rem if " %glue_high% " == " %task_name% " (
rem     set "eval_steps=500"
rem     set "prepruning_finetune_epochs=1"
rem     set "lagrangian_warmup_epochs=2"
rem )

set "eval_steps=500"
set "prepruning_finetune_epochs=1"
set "lagrangian_warmup_epochs=2"

set "pretrained_pruned_model=None"

rem FT after pruning
if " %pruning_type% " == " None " (
    set "pretrained_pruned_model=%10"
    set "learning_rate=%11"
    set "scheduler_type=none"
    set "output_dir=%pretrained_pruned_model%/FT-lr%learning_rate%"
    set "epochs=20"
    set "batch_size=64"
)

rem mkdir "%output_dir%"

python %code_dir%/run_glue_prune.py ^
       --output_dir %output_dir% ^
       --logging_steps %logging_steps% ^
       --task_name %task_name% ^
       --model_name_or_path %model_name_or_path% ^
       --ex_name %ex_name% ^
       --do_train ^
       --do_eval ^
       --max_seq_length %max_seq_length% ^
       --per_device_train_batch_size %batch_size% ^
       --per_device_eval_batch_size 32 ^
       --learning_rate %learning_rate% ^
       --reg_learning_rate %reg_learning_rate% ^
       --num_train_epochs %epochs% ^
       --overwrite_output_dir ^
       --save_steps %save_steps% ^
       --eval_steps %eval_steps% ^
       --evaluation_strategy steps ^
       --seed %seed% ^
       --pruning_type %pruning_type% ^
       --pretrained_pruned_model %pretrained_pruned_model% ^
       --target_sparsity %target_sparsity% ^
       --freeze_embeddings ^
       --do_distill ^
       --do_layer_distill ^
       --distillation_path %distillation_path% ^
       --distill_ce_loss_alpha %distill_ce_loss_alpha% ^
       --distill_loss_alpha %distill_layer_loss_alpha% ^
       --distill_temp %distill_temp% ^
       --scheduler_type %scheduler_type% ^
       --layer_distill_version %layer_distill_version% ^
       --prepruning_finetune_epochs %prepruning_finetune_epochs% 2>&1
        rem > %output_dir%\log.txt
