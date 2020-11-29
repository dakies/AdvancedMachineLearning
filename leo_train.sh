module load python_gpu/3.7.1 hdf5/1.10.1
pip3 install --user -r requirements.txt
bsub -n4 -W 239 -R "rusage[mem=4000,ngpus_excl_p=0]" -o "job_outputs/%J.lsf" python3 extract_features_v5.py "$@"

