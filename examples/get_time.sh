#num_passes=2
#for pass in $(seq 0 $(expr $num_passes - 1))
#do
#    python bert.py --result_file bert_time.txt
#    cat timer.txt >> bert_stats.txt
#done

num_passes=2
for pass in $(seq 0 $(expr $num_passes - 1))
do
    python resnext50.py --result_file resnext50_time.txt
    cat timer.txt >> resnext50_stats.txt
done