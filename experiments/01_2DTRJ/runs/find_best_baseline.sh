(
for d in $(\ls -d results/linear*baseline*L$1*); do
    echo -n $d"    "
    cat $d/HamiltonianSuccess.dat | awk -F',' '{for(i=1;i<=NF;i++) t+=$i; print t,(NF-1); t=0}'
done
) | sort -k2,2n 
