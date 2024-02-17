tol=(0.05 0.1 0.2)
delta=(0.125 0.25 0.5 1.0 2.0)
FEATURE="circuitousness"

for t in "${tol[@]}"
do 
    for d in "${delta[@]}"
    do
        python corpus/filter_attribute.py $FEATURE $d $t
    done
done