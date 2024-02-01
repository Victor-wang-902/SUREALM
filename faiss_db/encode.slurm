#!/bin/bash
#SBATCH --nodes=1
##SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --job-name=indexing
#SBATCH --mail-type=END
#SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
#SBATCH --output=index_%j_sent_suf.out
#SBATCH --error=index_%j_sent_suf.err
## #python fast_encoder.py --data_folder wikidata --out_folder fast_processed_wikidata_14 --cpu --for_fast
## #python fast_encoder.py --data_folder dummy_data --out_folder fast_dummy_processed_data --cpu --for_fast --local_pooling
## #python fast_encoder.py --data_folder wikidata/wikitext-103_12 --out_folder fast_processed_wikidata_12 --cpu --for_fast
## #python fast_encoder.py --data_folder processed_data_new --data_config data_config_topk.json --out_folder fast_processed_data_13_topk_more --cpu --resume_fast
## #python fast_encoder.py --model_name sentence-transformers/all-roberta-large-v1 --data_folder data --data_config data_config.json --out_folder processed_data_big --cpu
## #python fast_encoder.py --data_folder processed_data_new --data_config data_config_best.json --out_folder fast_processed_data_opt_multiqa --cpu --resume_fast
## #python fast_encoder.py --model_name sentence-transformers/all-MiniLM-L6-v2 --data_folder data --data_config data_config_best.json --out_folder processed_data_allmini --cpu --for_fast
## #python conditional_encoder.py --data_folder processed_data_new --data_config data_config_best.json --out_folder fast_processed_data_opt_dual_index --cpu --resume_fast --multi_index --interleave --index_config processed_data_new/index_config.json
## #python conditional_encoder.py --data_folder processed_data_new --data_config data_config_best.json --out_folder fast_processed_data_opt_dual_index --resume_fast --multi_index --interleave --index_config processed_data_new/index_config.json
## #python conditional_encoder.py --data_folder processed_data_new --data_config data_config_best.json --out_folder fast_processed_data_opt_dual_index_test_cpu --resume_fast --multi_index --interleave --index_config processed_data_new/index_config.json --cpu
## #python fast_encoder.py --data_folder processed_data_new --data_config data_config_window.json --out_folder fast_processed_data_13_window_more --cpu --resume_fast
## #python conditional_encoder.py --data_folder processed_data_new --data_config data_config_best.json --out_folder fast_processed_data_opt_dual_index_interleave --resume_fast --multi_index --interleave --index_config processed_data_new/index_config.json
## #python conditional_encoder.py --data_folder processed_data_new --data_config data_config_best.json --out_folder fast_processed_data_opt_dual_index --resume_fast --multi_index --interleave --index_config processed_data_new/index_config.json
## #python knowledge_encoder.py --data_folder data --data_config data_config_best.json --out_folder processed_data_knowledge_test
## #python knowledge_encoder.py --data_folder data --data_config data_config_best.json --out_folder processed_data_knowledge_test --knowledge_test
## #python conditional_encoder.py --model_name sentence-transformers/multi-qa-mpnet-base-cos-v1 --data_folder data --data_config data_config_best.json --out_folder fast_processed_data_multimp --for_fast
## #python conditional_encoder.py --model_name sentence-transformers/multi-qa-mpnet-base-cos-v1 --data_folder data --data_config data_config_best.json --out_folder fast_processed_data_multimp --for_fast

## #python knowledge_encoder.py --data_folder data --data_config data_config_best_test.json --out_folder processed_data_knowledge_test --knowledge_test
## #python knowledge_encoder.py --data_folder processed_data_knowledge_test --data_config data_config.json --out_folder fast_processed_data_knowledge_test --knowledge_test --resume_fast --index_config data/index_config_test.json

## #1.python conditional_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --data_folder data --data_config data_config_best.json --out_folder fast_processed_data_multi_distilbert --for_fast --not_preserve_position_embedding
##python conditional_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --data_folder data --data_config data_config_best.json --out_folder fast_processed_data_multi_distilroberta --for_fast
## #2.python knowledge_encoder.py --data_folder data --data_config data_config_best_test.json --out_folder processed_data_knowledge_test --knowledge_test
## #python knowledge_encoder.py --data_folder processed_data_knowledge_test --data_config data_config.json --out_folder fast_processed_data_knowledge_test --knowledge_test --resume_fast --index_config data/index_config_test.json
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-mpnet-base-cos-v1 --target_tokenizer gpt2 --data_folder data --data_config data_config_best.json --out_folder fast_processed_data_multi_mpnet_gpt2 --for_fast
## #python knowledge_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --target_tokenizer gpt2 --data_folder data --data_config data_config_best.json --out_folder fast_processed_data_all_roberta_gpt2 --for_fast

## #python knowledge_encoder.py --data_folder processed_data_all_roberta_gpt2 --data_config data_config.json --out_folder fast_processed_data_all_roberta_gpt2 --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_multi_distilroberta --data_config data_config.json --out_folder fast_processed_data_multi_distilroberta --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_multi_distilbert --data_config data_config.json --out_folder fast_processed_data_multi_distilbert --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_multi_mpnet_gpt2 --data_config data_config.json --out_folder fast_processed_data_multi_mpnet_gpt2 --resume_fast


## #python knowledge_encoder.py --data_folder processed_data_knowledge_test_multi --data_config data_config.json --out_folder fast_processed_data_knowledge_test --knowledge_test --resume_fast --index_config data/index_config_test.json --multi_index
## #python knowledge_encoder.py --data_folder processed_data_knowledge_test --data_config data_config.json --out_folder fast_processed_data_knowledge_test_multi --knowledge_test --resume_fast --index_config data/index_config_test.json --multi_index
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --target_tokenizer roberta-base --data_folder data --data_config data_config_best.json --out_folder fast_processed_data_distilbert_roberta --for_fast --not_preserve_position_embedding
## #python knowledge_encoder.py --data_folder processed_data_distilbert_roberta --data_config data_config.json --out_folder fast_processed_data_distilbert_roberta --resume_fast

## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --target_tokenizer roberta-base --data_folder data --data_config data_config_best.json --out_folder processed_data_distilbert_roberta_val --val_only --not_preserve_position_embedding
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --data_folder data --data_config data_config_best.json --out_folder processed_data_distilbert_val --val_only --not_preserve_position_embedding
## #python knowledge_encoder.py --data_folder data --data_config data_config_1gram.json --out_folder processed_data_130_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_2gram.json --out_folder processed_data_230_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_3gram.json --out_folder processed_data_330_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_2window.json --out_folder processed_data_132_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_4window.json --out_folder processed_data_134_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_6window.json --out_folder processed_data_136_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_8window.json --out_folder processed_data_138_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_16window.json --out_folder processed_data_1316_val --val_only
## #python knowledge_encoder.py --data_folder data --data_config data_config_20window.json --out_folder processed_data_1320_val --val_only

## #python knowledge_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --target_tokenizer gpt2 --data_folder data --data_config data_config_best.json --out_folder processed_data_distilroberta_gpt2_val --val_only 
## #python knowledge_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --data_folder data --data_config data_config_best.json --out_folder processed_data_distilroberta_val --val_only 
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-mpnet-base-cos-v1 --target_tokenizer gpt2 --data_folder data --data_config data_config_best.json --out_folder processed_data_mpnet_gpt2_val --val_only 

## #python knowledge_encoder.py --data_folder processed_data_130_val --data_config data_config.json --out_folder fast_processed_data_130_val --knowledge_test --index_config processed_data_130_val/index_config_130.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_132_val --data_config data_config.json --out_folder fast_processed_data_132_val --knowledge_test --index_config processed_data_132_val/index_config_132.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_134_val --data_config data_config.json --out_folder fast_processed_data_134_val --knowledge_test --index_config processed_data_134_val/index_config_134.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_136_val --data_config data_config.json --out_folder fast_processed_data_136_val --knowledge_test --index_config processed_data_136_val/index_config_136.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_138_val --data_config data_config.json --out_folder fast_processed_data_138_val --knowledge_test --index_config processed_data_138_val/index_config_138.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_1316_val --data_config data_config.json --out_folder fast_processed_data_1316_val --knowledge_test --index_config processed_data_1316_val/index_config_1316.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_1320_val --data_config data_config.json --out_folder fast_processed_data_1320_val --knowledge_test --index_config processed_data_1320_val/index_config_1320.json --resume_fast

## #python knowledge_encoder.py --data_folder processed_data_130_val --data_config data_config_110.json --out_folder fast_processed_data_110_val --knowledge_test --index_config processed_data_130_val/index_config_110.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_130_val --data_config data_config_120.json --out_folder fast_processed_data_120_val --knowledge_test --index_config processed_data_130_val/index_config_120.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_130_val --data_config data_config_140.json --out_folder fast_processed_data_140_val --knowledge_test --index_config processed_data_130_val/index_config_140.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_130_val --data_config data_config_180.json --out_folder fast_processed_data_180_val --knowledge_test --index_config processed_data_130_val/index_config_180.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_130_val --data_config data_config_1100.json --out_folder fast_processed_data_1100_val --knowledge_test --index_config processed_data_130_val/index_config_1100.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_130_val --data_config data_config_1160.json --out_folder fast_processed_data_1160_val --knowledge_test --index_config processed_data_130_val/index_config_1160.json --resume_fast

## #python knowledge_encoder.py --data_folder processed_data_230_val --data_config data_config.json --out_folder fast_processed_data_230_val --knowledge_test --index_config processed_data_230_val/index_config_230.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_330_val --data_config data_config.json --out_folder fast_processed_data_330_val --knowledge_test --index_config processed_data_330_val/index_config_330.json --resume_fast

## #python knowledge_encoder.py --data_folder data --data_config data_config_best.json --out_folder processed_data_multiqa_val --val_only 
## #python knowledge_encoder.py --model_name sentence-transformers/all-MiniLM-L6-v2 --data_folder data --data_config data_config_best.json --out_folder processed_data_allmini_val --val_only 

## #python knowledge_encoder.py --data_folder processed_data_distilbert_roberta_val --data_config data_config.json --out_folder fast_processed_data_distilbert_roberta_val --knowledge_test --index_config processed_data_distilbert_roberta_val/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_distilbert_val --data_config data_config.json --out_folder fast_processed_data_distilbert_val --knowledge_test --index_config processed_data_distilbert_val/index_config.json --resume_fast

## #python knowledge_encoder.py --data_folder processed_data_distilroberta_gpt2_val --data_config data_config.json --out_folder fast_processed_data_distilroberta_gpt2_val --knowledge_test --index_config processed_data_distilroberta_gpt2_val/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_distilroberta_val --data_config data_config.json --out_folder fast_processed_data_distilroberta_val --knowledge_test --index_config processed_data_distilroberta_val/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_mpnet_gpt2_val --data_config data_config.json --out_folder fast_processed_data_mpnet_gpt2_val --knowledge_test --index_config processed_data_mpnet_gpt2_val/index_config.json --resume_fast

## #python knowledge_encoder.py --data_folder processed_data_multiqa_val --data_config data_config.json --out_folder fast_processed_data_multiqa_val --knowledge_test --index_config processed_data_multiqa_val/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_allmini_val --data_config data_config.json --out_folder fast_processed_data_allmini_val --knowledge_test --index_config processed_data_allmini_val/index_config.json --resume_fast

## #python knowledge_encoder.py --data_folder processed_data_new --data_config data_config_best.json --out_folder fast_processed_data_opt_dual_index --resume_fast --multi_index --interleave --index_config processed_data_new/index_config.json

## #python knowledge_encoder.py --data_folder data --data_config data_config_10window.json --out_folder fast_processed_data_window_10 --for_fast
## #python knowledge_encoder.py --data_folder processed_data_1310_val --data_config data_config.json --out_folder fast_processed_data_1310_val --knowledge_test --index_config processed_data_1310_val/index_config_1310.json --resume_fast

## #python knowledge_encoder.py --data_folder dummy_data --data_config data_config.json --out_folder dummy_fast_processed_data_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_1gram.json --out_folder fast_processed_data_130_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_2window.json --out_folder fast_processed_data_132_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_4window.json --out_folder fast_processed_data_134_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_6window.json --out_folder fast_processed_data_136_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_8window.json --out_folder fast_processed_data_138_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_16window.json --out_folder fast_processed_data_1316_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_20window.json --out_folder fast_processed_data_1320_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_2gram.json --out_folder fast_processed_data_230_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_3gram.json --out_folder fast_processed_data_330_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_top1.json --out_folder fast_processed_data_110_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_top2.json --out_folder fast_processed_data_120_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_top4.json --out_folder fast_processed_data_140_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_top8.json --out_folder fast_processed_data_180_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_top10.json --out_folder fast_processed_data_1100_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_top16.json --out_folder fast_processed_data_1160_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_1gram.json --out_folder fast_processed_data_130_final --for_fast
## #python knowledge_encoder.py --data_folder data --data_config data_config_2window.json --out_folder fast_processed_data_132_final --for_fast

## #python knowledge_encoder.py --data_folder processed_data_136_final --data_config data_config.json --out_folder fast_processed_data_136_final --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_140_final --data_config data_config.json --out_folder fast_processed_data_140_final --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_1100_final --data_config data_config.json --out_folder fast_processed_data_1100_final --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_1316_final --data_config data_config.json --out_folder fast_processed_data_1316_final --resume_fast
## #python knowledge_encoder.py --data_folder processed_data_132_final --data_config data_config.json --out_folder fast_processed_data_132_final --resume_fast

## #python knowledge_encoder.py --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_multiqa_final_no_skip --for_fast --no_skip
## #python knowledge_encoder.py --model_name sentence-transformers/all-MiniLM-L6-v2 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_allmini_final --for_fast
## #python knowledge_encoder.py --model_name sentence-transformers/all-MiniLM-L6-v2 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_allmini_final_no_skip --for_fast --no_skip

## #python knowledge_encoder.py --data_folder data_final_test --data_config data_config_best_final.json --out_folder test_final_multiqa --val_only

## #python knowledge_encoder.py --data_folder data_final --data_config data_config_best_old.json --out_folder fast_processed_data_opt_multiqa_no_skip --for_fast --no_skip
## #python knowledge_encoder.py --data_folder test_final_multiqa --data_config data_config.json --out_folder fast_processed_data_multiqa_test --knowledge_test --index_config test_final_multiqa/index_config.json --resume_fast

## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --data_folder data_final_test --data_config data_config_best_old.json --out_folder processed_distilbert_test --val_only --not_preserve_position_embedding
## #python knowledge_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --target_tokenizer gpt2 --data_folder data_final_test --data_config data_config_best_old.json --out_folder processed_mpnet_gpt2_test --val_only
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-mpnet-base-cos-v1 --target_tokenizer gpt2 --data_folder data_final_test --data_config data_config_best_old.json --out_folder processed_roberta_gpt2_test --val_only

## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --target_tokenizer roberta-base --data_folder data_final_test --data_config data_config_best_old.json --out_folder processed_distilbert_roberta_test --val_only --not_preserve_position_embedding
## #python knowledge_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --data_folder data_final_test --data_config data_config_best_old.json --out_folder processed_distilroberta_test --val_only
## #python knowledge_encoder.py --model_name sentence-transformers/all-MiniLM-L6-v2 --data_folder data_final_test --data_config data_config_best_old.json --out_folder processed_allmini_test --val_only

## #python knowledge_encoder.py --data_folder processed_distilbert_test --data_config data_config.json --out_folder fast_processed_distilbert_test --knowledge_test --index_config processed_distilbert_test/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_mpnet_gpt2_test --data_config data_config.json --out_folder fast_processed_mpnet_gpt2_test --knowledge_test --index_config processed_mpnet_gpt2_test/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_roberta_gpt2_test --data_config data_config.json --out_folder fast_processed_roberta_gpt2_test --knowledge_test --index_config processed_roberta_gpt2_test/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_distilbert_roberta_test --data_config data_config.json --out_folder fast_processed_distilbert_roberta_test --knowledge_test --index_config processed_distilbert_roberta_test/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_distilroberta_test --data_config data_config.json --out_folder fast_processed_distilroberta_test --knowledge_test --index_config processed_distilroberta_test/index_config.json --resume_fast
## #python knowledge_encoder.py --data_folder processed_allmini_test --data_config data_config.json --out_folder fast_processed_allmini_test --knowledge_test --index_config processed_allmini_test/index_config.json --resume_fast

## #python knowledge_encoder.py --data_folder data_final --data_config data_config_best_old_test.json --out_folder processed_data_knowledge_test_corrected --knowledge_test
## #python knowledge_encoder.py --data_folder processed_data_knowledge_test_corrected --data_config data_config.json --out_folder fast_processed_data_knowledge_test_corrected --knowledge_test --resume_fast --index_config processed_data_knowledge_test_corrected/index_config.json --multi_index
## #python knowledge_encoder.py --data_folder processed_data_knowledge_test_corrected --data_config data_config.json --out_folder fast_processed_data_knowledge_test_corrected_interleave --knowledge_test --resume_fast --index_config processed_data_knowledge_test_corrected/index_config.json --multi_index --interleave

## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --target_tokenizer gpt2 --data_folder data_final --data_config data_config_best_old.json --out_folder fast_processed_data_multi_distilbert_gpt2 --for_fast --not_preserve_position_embedding


## #python knowledge_encoder.py --model_name sentence-transformers/all-MiniLM-L6-v2 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_minilm_final --for_fast
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_distilbert_final --for_fast --not_preserve_position_embedding
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --target_tokenizer gpt2 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_distilbert_gpt2_final --for_fast --not_preserve_position_embedding
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --target_tokenizer roberta-base --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_distilbert_roberta_final --for_fast --not_preserve_position_embedding
## #python knowledge_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_distilroberta_final --for_fast
## #python knowledge_encoder.py --model_name sentence-transformers/all-distilroberta-v1 --target_tokenizer gpt2 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_distilroberta_gpt2_final --for_fast
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-mpnet-base-cos-v1 --target_tokenizer gpt2 --data_folder data_final --data_config data_config_best_final.json --out_folder fast_processed_data_mpnet_gpt2_final --for_fast
## #python knowledge_encoder.py --data_folder processed_data_knowledge_test_corrected --data_config data_config.json --out_folder fast_processed_data_knowledge_test_corrected_knowledge_only --knowledge_test --resume_fast --index_config processed_data_knowledge_test_corrected/index_config_knowledge.json
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --target_tokenizer gpt2 --data_folder data_final --data_config data_config_best_old.json --out_folder fast_processed_data_multi_distilbert_gpt2 --for_fast --not_preserve_position_embedding
## #python knowledge_encoder.py --model_name sentence-transformers/multi-qa-distilbert-cos-v1 --data_folder data_final --data_config data_config_best_old_test.json --out_folder processed_data_knowledge_test_only_new --knowledge_test --not_preserve_position_embedding
## #python knowledge_encoder.py --data_folder processed_data_knowledge_test_only_new --data_config data_config.json --out_folder fast_processed_data_knowledge_test_only_new --knowledge_test --resume_fast --index_config data_final/index_config_only.json

## #python knowledge_encoder.py --model_name sentence-transformers/all-MiniLM-L12-v1 --data_folder data_final --data_config data_config_best_old.json --out_folder fast_processed_data_allminil12_final --for_fast

##python knowledge_encoder.py --model_name sentence-transformers/all-MiniLM-L12-v1 --data_folder data_final --data_config data_config_best_old.json --out_folder fast_processed_data_allminil12_final --for_fast
##python knowledge_encoder.py --data_folder processed_data_ --data_config data_config_best_old.json --out_folder fast_processed_data_all_multi_interleave --for_fast --index_config data_final/index_config.json --multi_index --interleave

##python knowledge_encoder.py --data_folder data_final --data_config data_config_default.json --out_folder fast_processed_data_default_sent_suffix1 --sent_suffix --for_fast
##python knowledge_encoder.py --data_folder fast_processed_data_default__suffix --data_config data_config.json --out_folder fast_processed_data_default_sent_suffix_final --sent_suffix --resume_fast

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ext3/miniconda3/envs/rblm/lib

singularity exec --nv \
--overlay /scratch/zw2374/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate rblm
python knowledge_encoder.py --data_folder data_final --data_config data_config_default.json --out_folder fast_processed_data_default --for_fast
"
