#!/bin/bash
#SBATCH --partition=electronic,hard
#SBATCH --job-name=TSPFN-Finetune
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi

### SBATCH --mem=120G
### SBATCH --cpus-per-task=12

ulimit -n 4096

#! ORCHID learning

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-FT-Transformer/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-FT-Transformer/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-FT-Transformer/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-FT-Transformer/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A2C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-TS/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=time-series
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-TS-A2C/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=time-series data.patients_kwargs.views=[A2C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-TS-A4C/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=time-series data.patients_kwargs.views=[A4C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-Tab/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-Tab-A2C/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular data.patients_kwargs.views=[A2C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-Tab-A4C/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular data.patients_kwargs.views=[A4C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-A2C/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A2C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-FTT-A4C/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C]

#! Asymetric fusion
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-DAFTED/seed${seed}' +experiment=orchid/xtab-interleaved exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment trainer.max_steps=4000
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-DAFTED/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-2UniFTs-TSinvert trainer.max_steps=4000
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-DAFTED-InfoNCE2/seed${seed}' +experiment=orchid/xtab-interpatient exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-2UniFTs-invert
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/IRENE-baseline/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=baseline-irene task.embed_dim=768

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-DAFTED/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-A2C/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment data.patients_kwargs.views=[A2C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-A4C/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment data.patients_kwargs.views=[A4C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-FrozeCross/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment task.model.encoder.n_cross_blocks=1
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-2UniFTs-NoSubmodule/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-2UniFTs-nosubmodule task.model.encoder.n_cross_blocks=1
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-2UniFTs/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-2UniFTs task.model.encoder.n_cross_blocks=1

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-DAFTED/seed${seed}' +experiment=orchid/xtab-interleaved exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-NoCross-A4C/seed${seed}' +experiment=orchid/xtab-interleaved exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment-nocross data.patients_kwargs.views=[A4C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-NoCrossFrozen-A4C/seed${seed}' +experiment=orchid/xtab-interleaved exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment-nocross data.patients_kwargs.views=[A4C] task.model.encoder.n_cross_blocks=1

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-DAFTED-TabPFNTokenizer/seed${seed}' +experiment=orchid/xtab-interleaved exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-ARTransformer/seed${seed}' +experiment=orchid/attribute-reg exclude_tabular_attrs=[diagnosis] seed=43 task/data=tabular+time-series
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-ARTransformer-Multi/seed${seed}' +experiment=orchid/attribute-reg exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series

#? From now on, DAFTED are no UNiFTs
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-DAFTED-A2COnly/seed${seed}' +experiment=orchid/xtab-alignment exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-interleaved-alignment data.patients_kwargs.views=[A2C]

#! TabPFN Unimodal

#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-Unimodal/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular task.model.fusion_module=null task.model.PFN_for_ts=False task.model.freeze_encoder=False
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-Unimodal-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular task.model.fusion_module=null task.model.PFN_for_ts=False task.model.freeze_encoder=True
#! poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Unimodal/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True
#! poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Unimodal-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-Unimodal/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular data.patients_kwargs.views=[A4C] task.model.fusion_module=null task.model.PFN_for_ts=False task.model.freeze_encoder=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-Unimodal-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular data.patients_kwargs.views=[A4C] task.model.fusion_module=null task.model.PFN_for_ts=False task.model.freeze_encoder=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-Unimodal-Frozen-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular data.patients_kwargs.views=[A4C] task.model.fusion_module=null task.model.PFN_for_ts=False task.model.freeze_encoder=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Unimodal/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn data.patients_kwargs.views=[A4C] task.model.fusion_module=null task.model.PFN_for_ts=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Unimodal-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn data.patients_kwargs.views=[A4C] task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True


#! TabPFN Multimodal

#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series task/model/fusion_module=mlp-fusion task.model.PFN_for_ts=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-TSPFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.PFN_for_ts=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-FrozenTSPFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-Learnable/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series task/model/fusion_module=mlp-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-Learnable-TSPFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-TSPFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-FrozenTSPFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-Learnable/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series task/model/fusion_module=dafted-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-Learnable-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-Learnable-TSPFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-L1O/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-L1O-SecondPart/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=False ckpt="/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-L1O/seed42/checkpoints/epoch\=1984-step\=1985.ckpt"
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DoublePFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion trainer.max_steps=2500
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DoublePFN-FrozenTSPFN/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.freeze_ts_pfn_encoder=True trainer.max_steps=2500
# for split in 0.1 0.2 0.3 0.6 0.7 0.8 0.9; do
#     poetry run didactic-runner "hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-Learnable-split${split}" +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.split_finetuning=$split +split=$split task.model.PFN_for_ts=False task.model.freeze_encoder=False
# done
# for test_batch_size in 1 4 16 32 64 128 256; do
#     poetry run didactic-runner "hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-TestBatchSize${test_batch_size}/seed${seed}" +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=False data.test_batch_size=$test_batch_size
# done
# poetry run didactic-runner "hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-TestBatchSize1/seed${seed}" +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=False data.test_batch_size=1

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-NoSecondaryEncoder/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.PFN_for_ts=False

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.PFN_for_ts=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.use_secondary_encoder=True task.model.PFN_for_ts=False
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-TSPFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.PFN_for_ts=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-FrozenTSPFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-Learnable-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=False
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-MLPFusion-Learnable-TSPFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=mlp-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=False
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-TSPFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-FrozenTSPFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-Learnable-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=False
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-Learnable-TSPFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.freeze_encoder=False task.model.PFN_for_ts=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DoublePFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion trainer.max_steps=2500
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DoublePFN-FrozenTSPFN-LoadedHead/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.freeze_ts_pfn_encoder=True trainer.max_steps=2500
# for split in 0.1 0.3 0.7 0.8 0.9; do
#     poetry run didactic-runner "hydra.run.dir=/data/stympopper/resOrchid/TabPFN-MultiModal-DAFusion-DoublePFN-split${split}" +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.split_finetuning=$split +split=$split task.model.PFN_for_ts=True
# done
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-DiffAtt-A4C/seed${seed}' +experiment=orchid/xtab-interleaved exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-diff-interleaved-alignment data.patients_kwargs.views=[A4C]
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/ORCHID-DAFTED-DiffAttFrozenCross-A4C/seed${seed}' +experiment=orchid/xtab-interleaved exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular+time-series task/model/encoder=xtab-diff-interleaved-alignment data.patients_kwargs.views=[A4C] task.model.encoder.n_cross_blocks=1
#! poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Learnable/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=tabular+time-series data.patients_kwargs.views=[A4C] task/model/fusion_module=dafted-fusion task.model.freeze_ts_pfn_encoder=True

#! Unimodal prediction with transformer
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-unimodal-TS/seed${seed}' +experiment=orchid/xtab exclude_tabular_attrs=[diagnosis] seed=42 task/data=time-series

#! Use LLMs

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/didacticWORKSHOP/ORCHID-TaBioBERT/seed${seed}' +experiment=orchid/bert exclude_tabular_attrs=[diagnosis] seed=42 task/data=tabular ~callbacks.model_checkpoint data.batch_size=64 trainer.max_steps=6000

#! Tabular Prediction below
# python /home/stympopper/didacticJerem/didactic/tasks/cardiac_records_prediction.py task/data=tabular ~task.time_series_attrs task.target_attr=diagnosis
# python /home/stympopper/didacticJerem/didactic/tasks/cardiac_records_prediction.py task/data=tabular ~task.time_series_attrs task.target_attr=diagnosis data.patients_kwargs.views=[A2C]
# python /home/stympopper/didacticJerem/didactic/tasks/cardiac_records_prediction.py task/data=tabular+time-series task.target_attr=diagnosis 

# for fold in 0 1 2 3 4; do
#     poetry run python /home/stympopper/didacticJerem/didactic/tasks/cardiac_records_prediction.py task/data=tabular ~task.time_series_attrs 'data.subsets.train=/data/stympopper/ORCHID/databasefolds/split_to_5/'$fold'/train.txt' 'data.subsets.val=/data/stympopper/ORCHID/databasefolds/split_to_5/'$fold'/val.txt' 'data.subsets.test=/data/stympopper/ORCHID/databasefolds/split_to_5/'$fold'/test.txt'
# done
#! DATA SPLITTING

# poetry run python dataprocessing/data/orchid/split_data.py --output_dir=/data/stympopper/ORCHID/database/ --stratify_attr=diagnosis --bins=5 --data_roots=/data/stympopper/ORCHID/database/
# poetry run python dataprocessing/data/orchid/split_data.py --output_dir=/data/stympopper/ORCHID/databaseA4C/ --stratify_attr=diagnosis --bins=5 --data_roots=/data/stympopper/ORCHID/database/ --views="A4C" --train_name=train_A4C  --test_name=test_A4C --test_size=0.15

#! DRAFT

# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Unimodal-Extraction/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Pretrained-Unimodal-Extraction-v2/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Pretrained-Unimodal-Extraction-v3/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v3.pt
#? poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Pretrained-Unimodal-FineTune-v3/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=False task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v3.pt
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Unimodal-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v2.pt
poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TabPFNv2-Unimodal-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True train=False predict=False data.test_batch_size=200
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFNv2-Unimodal-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2.pt
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-RoPE-Unimodal-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/pretrainingTSPFN/ckpts/TSPFN_v2-RoPE.pt predict=False data.test_batch_size=200
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-RoPE-zscoring/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE-zscoring.pt predict=False data.test_batch_size=200 task.positional_encoding=rope
poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-RoPE-zscoring-5Chans/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/pretrainingTSPFN/ckpts/TSPFN-RoPE-zscoring-5CHANS-nowarmup.pt predict=False data.test_batch_size=200 task.positional_encoding=rope
poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Baseline-zscoring/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/pretrainingTSPFN/ckpts/TSPFNFM_Baseline-zscoring.pt predict=False data.test_batch_size=200
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-Unimodal-Frozen-v3/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v3.pt
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-UnimodalCollapse-Frozen-v3/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v3.pt task.time_series_processing_option=collapse
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-UnimodalSinusoidal-Frozen-v3/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True train=False task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v3.pt task.time_series_positional_encoding=sinusoidal
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-UnimodalPELearned-Frozen-v3/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v3.pt task.time_series_positional_encoding=learned
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/TSPFN-FeatureFusion/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/tspfn_encoder_weights_v3.pt task.time_series_feature_fusion=True
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/SinTSPFN-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/sintspfn_encoder_weights.ckpt train=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/SinTSPFN-SubspaceSinusPE-Frozen/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=True task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/sintspfn_encoder_weights.ckpt task.time_series_positional_encoding=sinusoidal train=False
# poetry run didactic-runner 'hydra.run.dir=/data/stympopper/resOrchid/SinTSPFN-SubspaceSinusPE/seed${seed}' +experiment=orchid/tabpfn-multimodal seed=42 task/data=time-series-pfn task.model.fusion_module=null task.model.PFN_for_ts=True task.model.freeze_ts_pfn_encoder=False task.model.encoder.updated_pfn_path=/home/stympopper/didacticJerem/ckpts/sintspfn_encoder_weights.ckpt task.time_series_positional_encoding=sinusoidal