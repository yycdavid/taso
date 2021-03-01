num_passes=1
postfix=10

models=(
    squeezenet
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1))
    do
        python test_onnx.py --file "$model".onnx --result_file "$model"_time_"$postfix".txt --iter $postfix
        cat timer.txt >> "$model"_stats_"$postfix".txt
    done
done
