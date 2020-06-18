

sbatch='bash'

#Grid search:

#for L in 4 6 8 10 12 14 16; do
#    for dbeta in 0.0001 0.0002 0.0004 0.0008 0.001 0.002 0.004 0.008 0.01 0.02 0.04 0.08 0.2 0.4 0.8; do
#        $sbatch submit_baseline.s $L $dbeta
#    done
#done

#$sbatch submit_baseline.s 16 0.002
$sbatch submit_baseline.s 14 0.002
$sbatch submit_baseline.s 12 0.01
$sbatch submit_baseline.s 10 0.002
$sbatch submit_baseline.s 8 0.004
$sbatch submit_baseline.s 6 0.00010
$sbatch submit_baseline.s 4 0.002

