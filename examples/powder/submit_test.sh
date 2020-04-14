#!/bin/bash
#SBATCH --partition GPU-shared
#SBATCH --gres=gpu:k80:1
#SBATCH --nodes 1
#SBATCH --time 08:00:00 
#SBATCH --job-name='powder test'
#SBATCH --output=test_out.stdout
module load cuda/10.0
module load gcc/6.3.0
source ../../ampis_env/bin/activate 
echo $(which python)

ts=`date +%Y-%m-%d_%H-%M_%S` # timestamp
job_name="particles"
python_file="explore_data_powder.py"
echo "python file: ${python_file}"

olddir="${ts}_${job_name}"
newdir="../../../batch_job_results/${ts}_${job_name}"


python3 "${python_file}"



mkdir $olddir
mv output $olddir
cp ./submit_test.sh $olddir
mv test_out.stdout $olddir
cp $python_file $olddir

#mkdir $newdir

#mv $olddir $newdir 
