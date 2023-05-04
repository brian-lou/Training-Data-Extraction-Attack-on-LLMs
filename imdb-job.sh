#!/bin/bash
#SBATCH --job-name=gpt-2  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1      # number of tasks per nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=blou@princeton.edu

module purge
module load anaconda3/2021.5
conda activate nlp

python gpt-2-imdb_extraction.py --N 20000 --batch-size 10 --internet-sampling
# python llama-extraction2.py --N 15000 --internet-sampling --wet-file commoncrawl.warc.wet 
# python extraction.py --N 20000 --batch-size 10 --internet-sampling --wet-file commoncrawl.warc.wet 