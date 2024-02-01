#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --job-name=training
#SBATCH --mail-type=END
#SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
#SBATCH --output=train_%j_hyper_old.out
#SBATCH --error=train_%j_hyper_old.err

## #python train_script.py --model_path v6/120000 --data_folder processed_data_w_eval --from_checkpoint --freeze_backbone --output v6_cls --epochs 100 --save_head
## #python train_script.py --model_path v6_base/120000 --data_folder processed_data_w_eval --from_checkpoint --freeze_backbone --output v6_cls_base --epochs 100 --save_head --not_retrieval --not_cross_attention
## #python train_script.py --model_path v6_cross_not_init/120000 --data_folder processed_data_w_eval --from_checkpoint --freeze_backbone --output v6_cls_cross_not_init --epochs 100 --save_head
## #python train_script.py --model_path v6_simple/120000 --data_folder processed_data_w_eval --from_checkpoint --freeze_backbone --output v6_cls_simple --epochs 100 --save_head --not_cross_attention
## #python train_script.py --model_path v7_base/120000 --data_folder processed_data_w_eval --from_checkpoint --freeze_backbone --output v7_cls_base --epochs 100 --save_head --not_retrieval --not_cross_attention
## #python train_script.py --model_path v7_simple_shuffled/120000 --data_folder processed_data_w_eval --from_checkpoint --freeze_backbone --output v7_cls_simple --epochs 100 --save_head --not_cross_attention
## #python train_script.py --data_folder fast_processed_data  --output v9_test --epochs 5 --save_head --external_embedding --save_epochs 5 --no_gpu
## #python train_script.py --data_folder fast_processed_data --model_path v8/30 --from_checkpoint --output v8_eval --external_embedding --no_gpu --eval_only --batch_size 1
## #python train_script.py --data_folder fast_processed_data  --output v9 --epochs 100 --save_head --external_embedding --save_epochs 1
## #python train_script.py --data_folder fast_processed_data  --output v9_base --epochs 100 --save_head --not_retrieval --not_cross_attention --save_epochs 1
## #python train_script.py --data_folder fast_processed_data  --output v10 --epochs 100 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --data_folder fast_processed_data  --output v10_base --epochs 100 --save_head  --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path v10/35 --data_folder fast_processed_data  --output v10_eval --not_fast --external_embedding --from_checkpoint --eval_only --online_eval --eval_topk 4 --eval_ngram 2
## #python train_script.py --model_path v10/35 --data_folder fast_processed_data  --output v10_eval --not_fast --external_embedding --from_checkpoint --eval_only --topk 4
## #python train_script.py --model_path v10/35 --data_folder fast_processed_data  --output v10_eval --not_fast --external_embedding --from_checkpoint --eval_only --online_eval --eval_topk 5 --eval_ngram 2
## #python train_script.py --model_path v10/35 --data_folder dummy_processed_data_w_eval  --output v10_eval --not_fast --external_embedding --from_checkpoint --eval_only --online_eval --eval_topk 5 --eval_ngram 2
## #python train_script.py --model_path v10/35 --data_folder fast_processed_data  --output v10_eval --not_fast --external_embedding --from_checkpoint --eval_only --online_eval --eval_topk 3 --eval_ngram 3
## #python train_script.py --data_folder fast_processed_data/13 --batch_size 128 --output std_13_quad_gpu128_200epochs --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --data_folder fast_processed_data/13  --output base_quad_gpu128_200epochs --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_folder fast_processed_data/13  --output std_13_quad_gpu128_200epochs_weight_init --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_folder fast_processed_data/13  --output base_quad_gpu128_200epochs_weight_init --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_folder fast_processed_data/13  --output std_13_quad_gpu128_200epochs_weight_init_pos_windowed_2 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --
## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_folder fast_processed_data_13 --output small_baseline_base --batch_size 128 --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --from_scratch --data_folder fast_processed_data_13 --output small_baseline_from_scratch --batch_size 128 --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention"
## #python train_script.py --model_path bert-base-uncased --data_folder fast_processed_data_13 --output big_baseline_base --batch_size 128 --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --from_scratch --model_path sentence-transformers/all-roberta-large-v1 --data_folder fast_processed_data_13 --output big_baseline_from_scratch --batch_size 128 --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path roberta-large --data_folder fast_processed_data_13 --output big_baseline_roberta_large --batch_size 128 --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config_1.json --data_folder fast_processed_data_topk  --output small_topk_1 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config_0.json --data_folder fast_processed_data_window  --output small_window_0 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script_alt.py --from_scratch --model_path sentence-transformers/all-roberta-large-v1 --data_folder fast_processed_data_13 --output big_baseline_from_scratch --batch_size 8 --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_topk_10  --output small_window_10 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini  --output small_allmini_minilm --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_config data_config.json --data_folder fast_processed_data_opt_allmini  --output small_allmini_allmini --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa  --output small_multiqa_minilm --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa  --output small_multiqa_multiqa --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script_new.py --model_path gpt2 --data_folder fast_processed_data_13 --output big_baseline_gpt2_test --batch_size 128 --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_topk_10  --output small_topk_10 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config_20.json --data_folder fast_processed_data_window_more  --output small_window_20 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding


## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_multi_distilroberta  --output large_distilroberta_roberta --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path bert-base-uncased --data_config data_config.json --data_folder fast_processed_data_multi_distilbert  --output large_distilbert_bert --batch_size 64 --epochs 200 --save_head  --save_epochs 1 --external_embedding

## #python train_script.py --model_path bert-base-uncased --data_folder fast_processed_data_13 --output big_baseline_base_64_low --batch_size 64 --epochs 200 --lr 1e-5 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path bert-base-uncased --data_config data_config.json --data_folder fast_processed_data_multi_distilbert  --output large_distilbert_bert_low --batch_size 64 --epochs 200 --save_head --lr 1e-5  --save_epochs 1 --external_embedding

## #python train_script.py --model_path gpt2 --data_folder fast_processed_data_13 --output large_baseline_gpt2_test --batch_size 64 --epochs 200 --save_head --lr 1e-5 --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_multi_distilroberta  --output large_distilroberta_roberta_64_low --batch_size 64 --lr 1e-5 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path roberta-base --data_folder fast_processed_data_13 --output big_baseline_roberta_64_low --batch_size 64 --epochs 200 --lr 1e-5 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --from_scratch --model_path roberta-base --data_folder fast_processed_data_13 --output big_baseline_roberta_scratch_64_low --batch_size 64 --epochs 200 --lr 1e-5 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention
## #python train_script.py --from_scratch --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_multi_distilroberta  --output large_distilroberta_roberta_from_scratch_64_low --batch_size 64 --lr 1e-5 --epochs 200 --save_head  --save_epochs 1 --external_embedding

## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_distilbert_roberta  --output large_distilbert_roberta_64_low --batch_size 64 --lr 1e-5 --epochs 200 --save_head  --save_epochs 1 --external_embedding

## #echo 1310
## #python train_script.py --model_path small_window_0 --data_config data_config.json --data_folder fast_processed_data_130_val  --output 130_val --external_embedding --eval_only
## #echo 110
## #python validation_script.py --model_path small_topk_1 --data_config data_config.json --data_folder fast_processed_data_110_val  --output 110_val --external_embedding --eval_only
## #echo 120
## #python validation_script.py --model_path small_topk_2 --data_config data_config.json --data_folder fast_processed_data_120_val  --output 120_val --external_embedding --eval_only
## #echo 140
## #python validation_script.py --model_path small_topk_4 --data_config data_config.json --data_folder fast_processed_data_140_val  --output 140_val --external_embedding --eval_only
## #echo 180
## #python validation_script.py --model_path small_topk_8 --data_config data_config.json --data_folder fast_processed_data_180_val  --output 180_val --external_embedding --eval_only
## #echo 1100
## #python validation_script.py --model_path small_topk_10 --data_config data_config.json --data_folder fast_processed_data_1100_val  --output 1100_val --external_embedding --eval_only
## #echo 1160
## #python validation_script.py --model_path small_topk_16 --data_config data_config.json --data_folder fast_processed_data_1160_val  --output 1160_val --external_embedding --eval_only
## #echo 130
## #python validation_script.py --model_path small_window_0 --data_config data_config.json --data_folder fast_processed_data_130_val  --output 130_val --external_embedding --eval_only
## #echo 132
## #python validation_script.py --model_path small_window_2 --data_config data_config.json --data_folder fast_processed_data_132_val  --output 132_val --external_embedding --eval_only
## #echo 134
## #python validation_script.py --model_path small_window_4 --data_config data_config.json --data_folder fast_processed_data_134_val  --output 134_val --external_embedding --eval_only
## #echo 136
## #python validation_script.py --model_path small_window_6 --data_config data_config.json --data_folder fast_processed_data_136_val  --output 136_val --external_embedding --eval_only
## #echo 138
## #python validation_script.py --model_path small_window_8 --data_config data_config.json --data_folder fast_processed_data_138_val  --output 138_val --external_embedding --eval_only
## #echo 1310
## #python validation_script.py --model_path small_window_10 --data_config data_config.json --data_folder fast_processed_data_1310_val  --output 1310_val --external_embedding --eval_only
## #echo 1316
## #python validation_script.py --model_path small_window_16 --data_config data_config.json --data_folder fast_processed_data_1316_val  --output 1316_val --external_embedding --eval_only
## #echo 1320
## #python validation_script.py --model_path small_window_20 --data_config data_config.json --data_folder fast_processed_data_1320_val  --output 1320_val --external_embedding --eval_only
## #echo 230
## #python validation_script.py --model_path std_23_quad_gpu --data_config data_config.json --data_folder fast_processed_data_230_val  --output 230_val --external_embedding --eval_only
## #echo 330
## #python validation_script.py --model_path std_33_quad_gpu --data_config data_config.json --data_folder fast_processed_data_330_val  --output 330_val --external_embedding --eval_only

## #echo allmini-allmini
## #python validation_script.py --model_path small_allmini_allmini --data_config data_config.json --data_folder fast_processed_data_allmini_val  --output allmini_allmini_val --external_embedding --eval_only
## #echo allmini-minilm
## #python validation_script.py --model_path small_allmini_minilm --data_config data_config.json --data_folder fast_processed_data_allmini_val  --output allmini_minilm_val --external_embedding --eval_only
## #echo multiqa-minilm
## #python validation_script.py --model_path small_multiqa_minilm --data_config data_config.json --data_folder fast_processed_data_multiqa_val  --output multiqa_minilm_val --external_embedding --eval_only
## #echo multiqa-multiqa
## #python validation_script.py --model_path small_multiqa_multiqa --data_config data_config.json --data_folder fast_processed_data_multiqa_val  --output multiqa_multiqa_val --external_embedding --eval_only
## #echo distilbert-roberta
## #python validation_script.py --model_path large_distilbert_roberta_64_low --data_config data_config.json --data_folder fast_processed_data_distilbert_roberta_val  --output distilbert_roberta_val --external_embedding --eval_only
## #echo distilroberta-roberta
## #python validation_script.py --model_path large_distilroberta_roberta_64_low --data_config data_config.json --data_folder fast_processed_data_distilroberta_val  --output distilroberta_roberta_val --external_embedding --eval_only
## #echo distilbert-bert
## #python validation_script.py --model_path large_distilbert_bert_low --data_config data_config.json --data_folder fast_processed_data_distilbert_val  --output distilbert_bert_val --external_embedding --eval_only

## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_110_final  --output small_val_110 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_120_final  --output small_val_120 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_130_final  --output small_val_130 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_134_final  --output small_val_134 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_138_final  --output small_val_138 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_140_final  --output small_val_140 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_180_final  --output small_val_180 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_230_final  --output small_val_230 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_330_final  --output small_val_330 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_1160_final  --output small_val_1160 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_1320_final  --output small_val_1320 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_136_final  --output small_val_136 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_1316_final  --output small_val_1316 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_1100_final  --output small_val_1100 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_window_10  --output small_val_1310 --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding

## #python validation_script.py --model_path small_val_1310 --data_config data_config.json --data_folder fast_processed_data_1310_val --output 1310_val --external_embedding --eval_only

## #echo bert
## #python validation_script.py --model_path big_baseline_base_64_low --data_config data_config.json --data_folder data_for_baseline  --output bert_baseline_val --external_embedding --test_eval --not_retrieval --not_cross_attention
## #echo roberta
## #python validation_script.py --model_path big_baseline_roberta_64_low --data_config data_config.json --data_folder data_for_baseline  --output roberta_baseline_val --external_embedding --test_eval --not_retrieval --not_cross_attention
## #echo gpt2
## #python validation_script.py --model_path large_baseline_gpt2_test --data_config data_config.json --data_folder data_for_baseline  --output gpt2_baseline_val --external_embedding --test_eval --not_retrieval --not_cross_attention
## #echo minilm
## #python validation_script.py --model_path small_baseline_base --data_config data_config.json --data_folder data_for_baseline  --output minilm_baseline_val --external_embedding --test_eval --not_retrieval --not_cross_attention
## #echo minilm from scratch
## #python validation_script.py --model_path small_baseline_from_scratch --data_config data_config.json --data_folder data_for_baseline  --output minilm_baseline_from_scratch_val --external_embedding --test_eval --not_retrieval --not_cross_attention

## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_no_skip  --output small_no_skip --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_multi_mpnet_gpt2  --output large_mp_gpt2 --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_all_roberta_gpt2  --output large_roberta_gpt2 --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path bert-base-uncased --data_config data_config.json --data_folder fast_processed_data_multi_distilbert  --output large_distilbert_bert --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected  --output small_multiqa_multiqa_corrected --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected  --output small_multiqa_minilm_corrected --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_138_final  --output small_multiqa_minilm_final --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --from_scratch --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected  --output small_multiqa_minilm_corrected_from_scratch --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python validation_script.py --model_path small_allmini_allmini --data_config data_config.json --data_folder fast_processed_allmini_test --output allmini_allmini_test --external_embedding --eval_only
## #python validation_script.py --model_path small_allmini_minilm --data_config data_config.json --data_folder fast_processed_allmini_test --output allmini_minilm_test --external_embedding --eval_only
## #python validation_script.py --model_path large_distilbert_roberta_64_low --data_config data_config.json --data_folder fast_processed_distilbert_roberta_test --output bert_roberta_test --external_embedding --eval_only
## #python validation_script.py --model_path large_distilroberta_roberta_64_low --data_config data_config.json --data_folder fast_processed_distilroberta_test --output roberta_roberta_test --external_embedding --eval_only

## #echo multiqa-minilm not interleave
## #python validation_script.py --model_path small_multiqa_minilm_corrected --data_config data_config.json --data_folder fast_processed_data_knowledge_test_corrected --output multiqa_allmini_test --external_embedding --eval_only
## #echo multiqa-minilm interleave
## #python validation_script.py --model_path small_multiqa_minilm_corrected --data_config data_config.json --data_folder fast_processed_data_knowledge_test_corrected_interleave --output multiqa_allmini_test_interleave --external_embedding --eval_only

## #echo baseline
## #python validation_script.py --model_path small_baseline_base --data_config data_config_ood.json --data_folder data_for_baseline --output baseline_allmini_test_ood --external_embedding --eval_only --not_retrieval --not_cross_attention
## #echo multiqa-minilm
## #python validation_script.py --model_path small_multiqa_minilm_corrected --data_config data_config.json --data_folder fast_processed_data_knowledge_test_corrected_interleave_ood --output multiqa_allmini_test_ood --external_embedding --eval_only

## #python validation_script.py --model_path small_multiqa_minilm_corrected --data_config data_config.json --data_folder fast_processed_data_knowledge_test_corrected_knowledge_only --output multiqa_allmini_test_knowledge_only --external_embedding --eval_only

## #python train_script.py --from_scratch --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini  --output small_allmini_minilm_from_scratch --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_allmini_final  --output small_allmini_minilm_final --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --from_scratch --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_allmini_final  --output small_allmini_minilm_from_scratch_final --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_config data_config.json --data_folder fast_processed_data_allmini_final --output small_allmini_allmini_final --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_138_final  --output small_multiqa_minilm_final --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --from_scratch --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_138_final  --output small_multiqa_minilm_from_scratch_final --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python train_script.py --model_path bert-base-uncased --data_config data_config.json --data_folder fast_processed_data_distilbert_final --output large_distilbert_bert_final --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_distilbert_roberta_final --output large_distilbert_roberta_final --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_distilbert_gpt2_final --output large_distilbert_gpt2_final --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_distilroberta_final --output large_distilroberta_roberta_final --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_distilroberta_gpt2_final --output large_distilroberta_gpt2_final --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_multi_distilbert_gpt2 --output large_distilbert_gpt2 --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval

## python train_script.py --from_scratch --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output small_allmini_minilm_corrected_from_scratch --batch_size 128 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval
## python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_multi_distilbert_gpt2 --output large_distilbert_gpt2_not_cross --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention
## python train_script.py --model_path  nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output small_allmini_minilm_not_cross  --epochs 200  --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention
## #python train_script.py --model_path  roberta-base --data_config data_config.json --data_folder fast_processed_data_distilbert_roberta --output large_distilbert_roberta_not_cross --batch_size 64 --lr 1e-5  --epochs 200  --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention
## python train_script.py --model_path  roberta-base --data_config data_config.json --data_folder fast_processed_data_multi_distilroberta --output large_distilroberta_roberta_not_cross --batch_size 64 --lr 1e-5  --epochs 200  --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention
## #python train_script.py --from_scratch --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output small_multiqa_minilm_from_scratch_not_cross  --epochs 200  --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention
## #python train_script.py --model_path bert-base-uncased --data_config data_config.json --data_folder fast_processed_data_multi_distilbert --output large_distilbert_bert_not_cross  --epochs 200  --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention --batch_size 64 --lr 1e-6
## #python train_script.py --model_path  --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output small_multiqa_minilm_not_cross  --epochs 200  --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output small_allmini_allmini_not_cross  --epochs 200  --save_head  --save_epochs 1 --external_embedding --test_eval --not_cross_attention
## #python train_script.py --model_path large_distilbert_bert_retrained/18 --data_config data_config.json --data_folder fast_processed_data_multi_distilbert  --output distilbert_bert_analysis --external_embedding --debug --eval_only --batch_size 1
## #python validation_script.py --model_path large_distilbert_bert_retrained --data_config data_config.json --data_folder fast_processed_data_knowledge_test_corrected_mixed --output large_distilbert_bert_mixed_test  --external_embedding --eval_only
## #python validation_script.py --model_path big_baseline_base_64_low --data_config data_config_ood.json --data_folder data_for_baseline --output big_baseline_mixed_test --external_embedding --eval_only --not_retrieval --not_cross_attention
## #python validation_script.py --model_path large_distilbert_bert_retrained --data_config data_config.json --data_folder fast_processed_data_knowledge_test_only_new --output large_distilbert_bert_test_only_new  --external_embedding --eval_only

## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_folder data_for_baseline --output allminil6_baseline --epochs 200 --save_head --save_epochs 1 --external_embedding --not_retrieval --not_cross_attention

## #python validation_script.py --from_checkpoint --model_path multiqa_l6_baseline  --data_folder data_for_baseline --output minilm_l6_baseline_eval --external_embedding --not_retrieval --not_cross_attention --eval_only --test_eval

##python validation_script.py --from_checkpoint --model_path minilm_l12_baseline  --data_folder data_for_baseline --output minilm_l12_baseline_eval --external_embedding --not_retrieval --not_cross_attention --eval_only --test_eval
##python validation_script.py --from_checkpoint --model_path allmini_l12_baseline  --data_folder data_for_baseline --output minilm_l12_baseline_eval --external_embedding --not_retrieval --not_cross_attention --eval_only --test_eval
##python validation_script.py --from_checkpoint --model_path allmini_l6_baseline  --data_folder data_for_baseline --output minilm_l12_baseline_eval --external_embedding --not_retrieval --not_cross_attention --eval_only --test_eval

## #python train_script.py --model_path  --data_config data_config.json --data_folder fast_processed_data_distilroberta_final --output large_distilroberta_roberta_final --batch_size 64 --epochs 200 --lr 1e-5 --save_head  --save_epochs 1 --external_embedding --test_eval
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_config data_config.json --data_folder fast_processed_data_allminil12_final --output allminil12_allminilml12 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval

## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multiqal6_minilml6_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multiqal6_multiqal6_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll6_alll6_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path microsoft/MiniLM-L12-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll6_minil12_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path microsoft/MiniLM-L12-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multil6_minil12_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path microsoft/MiniLM-L12-H384-uncased --data_config data_config.json --data_folder fast_processed_data_allminil12_final --output alll12_minil12_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L12-v1 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll6_alll12_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L12-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multil6_alll12_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L12-v1 --data_config data_config.json --data_folder fast_processed_data_allminil12_final --output alll12_alll12_not_concat --epochs 50 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self

## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multiqal6_minilml6_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multiqal6_multiqal6_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll6_alll6_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path microsoft/MiniLM-L12-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll6_minil12_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path microsoft/MiniLM-L12-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multil6_minil12_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path microsoft/MiniLM-L12-H384-uncased --data_config data_config.json --data_folder fast_processed_data_allminil12_final --output alll12_minil12_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L12-v1 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll6_alll12_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L12-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multil6_alll12_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L12-v1 --data_config data_config.json --data_folder fast_processed_data_allminil12_final --output alll12_alll12_not_concat_100e --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self

## #python train_script.py --model_path bert-base-uncased --data_config data_config.json --data_folder fast_processed_data_multi_distilbert --output distilbert_bert_not_concat --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self

## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_multi_distilroberta --output distilroberta_roberta_not_concat --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_distilbert_roberta --output distilbert_roberta_not_concat --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_multi_mpnet_gpt2 --output mpnet_gpt2_not_concat --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_multi_distilbert_gpt2 --output distilbert_gpt2_not_concat --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
## #python train_script.py --model_path gpt2 --data_config data_config.json --data_folder fast_processed_data_all_roberta_gpt2 --output roberta_gpt2_not_concat --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self

## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_opt_multiqa_corrected --output multiqal6_multiqal6_not_concat_100e_128 --epochs 100 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self --batch_size 128

## #python train_script.py --model_path nreimers/MiniLM-L6-H384-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output minil6_fair_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --fair_baseline --not_retrieval --not_cross_attention --batch_size 128
## #python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output multil6_fair_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --fair_baseline --not_retrieval --not_cross_attention --batch_size 128
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L6-v2 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll6_fair_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --fair_baseline --not_retrieval --not_cross_attention --batch_size 128
## #python train_script.py --model_path sentence-transformers/all-MiniLM-L12-v1 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output alll12_fair_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --fair_baseline --not_retrieval --not_cross_attention --batch_size 128
## #python train_script.py --model_path bert-base-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output bert_fair_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --fair_baseline --not_retrieval --not_cross_attention --lr 1e-5
## #python train_script.py --model_path roberta-base --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output roberta_fair_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --fair_baseline --not_retrieval --not_cross_attention --lr 1e-5

## #python train_script.py --model_path distilroberta-base --data_config data_config.json --data_folder fast_processed_data_multi_distilroberta --output distilroberta_distilroberta_not_concat --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self --lr 1e-5

## #python train_script.py --model_path distilgpt2 --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output distilgpt2_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_retrieval --not_cross_attention --lr 1e-5

## #python train_script.py --model_path distilgpt2 --data_config data_config.json --data_folder fast_processed_data_multi_distilbert_gpt2 --output distilbert_distilgpt2 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --lr 1e-5

##python train_script.py --model_path distilbert-base-uncased --data_config data_config.json --data_folder fast_processed_data_opt_allmini --output distilbert_baseline --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_retrieval --not_cross_attention --lr 1e-5

##python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder /scratch/zw2374/public/faiss_db/fast_processed_data/13  --output full_sent_suffix_model_old --batch_size 128 --lr 2e-5 --epochs 200 --save_head  --save_epochs 1 --external_embedding --not_concat_self

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ext3/miniconda3/envs/rblm/lib

singularity exec --nv \
--overlay /scratch/zw2374/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate rblm
python train_script.py --model_path sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --data_config data_config.json --data_folder /scratch/zw2374/public/faiss_db/fast_processed_data_default  --output suffix_suffix_model --batch_size 128 --lr 2e-5 --epochs 200 --save_head  --save_epochs 1 --external_embedding --test_eval --not_concat_self
"
