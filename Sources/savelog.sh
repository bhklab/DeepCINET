#!/bin/bash
output=${1}.out
errorfile=${1}.err
cp ${output} tmp.out
sed -i 's/\r/\n/g' tmp.out
cat tmp.out ${errorfile} > log/${2}.out
rm tmp.out
