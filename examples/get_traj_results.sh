num_passes=5
postfix=10

models=(
    inceptionv3
    resnext50
    bert
    nasrnn
    nasnet_a
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1))
    do
        python "$model".py --result_file "$model"_time_"$postfix".txt --iter $postfix
        cat timer.txt >> "$model"_stats_"$postfix".txt
    done
done

models=(
    squeezenet
    vgg19-7
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1))
    do
        python test_onnx.py --file "$model".onnx --result_file "$model"_time_"$postfix".txt --iter $postfix
        cat timer.txt >> "$model"_stats_"$postfix".txt
    done
done


num_passes=5
postfix=30

models=(
    inceptionv3
    resnext50
    bert
    nasrnn
    nasnet_a
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1))
    do
        python "$model".py --result_file "$model"_time_"$postfix".txt --iter $postfix
        cat timer.txt >> "$model"_stats_"$postfix".txt
    done
done

models=(
    squeezenet
    vgg19-7
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1))
    do
        python test_onnx.py --file "$model".onnx --result_file "$model"_time_"$postfix".txt --iter $postfix
        cat timer.txt >> "$model"_stats_"$postfix".txt
    done
done


num_passes=5
postfix=100

models=(
    inceptionv3
    resnext50
    bert
    nasrnn
    nasnet_a
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1))
    do
        python "$model".py --result_file "$model"_time_"$postfix".txt --iter $postfix
        cat timer.txt >> "$model"_stats_"$postfix".txt
    done
done

models=(
    squeezenet
    vgg19-7
)

for model in "${models[@]}"; do
    for pass in $(seq 0 $(expr $num_passes - 1))
    do
        python test_onnx.py --file "$model".onnx --result_file "$model"_time_"$postfix".txt --iter $postfix
        cat timer.txt >> "$model"_stats_"$postfix".txt
    done
done
