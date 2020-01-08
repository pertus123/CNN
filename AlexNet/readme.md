#cifar10 - batch_size = 64 EPOCH 10회 수행

#sequential 안에서 순차적으로 정의할 때, 
#기존의 호출 방법에 비해 느림.

구글 colab
OS : Ubuntu 18.04.3 LTS
CPU : 
processor 0, Intel(R) Xeon(R) CPU @ 2.00GHz
processor 1, Intel(R) Xeon(R) CPU @ 2.00GHz
Memory Total:       13335180 kB, 13GB
Driver Version: 418.67       CUDA Version: 10.1 

[epoch 1,imgs  6400] loss: 2.7832706 time : 10.778 s
[epoch 1,imgs 12800] loss: 2.2006576 time : 10.851 s
[epoch 1,imgs 19200] loss: 1.9440229 time : 10.749 s
[epoch 1,imgs 25600] loss: 1.7587621 time : 10.650 s
[epoch 1,imgs 32000] loss: 1.6501025 time : 10.558 s
[epoch 1,imgs 38400] loss: 1.5211828 time : 10.452 s
[epoch 1,imgs 44800] loss: 1.4141591 time : 10.507 s
[epoch 2,imgs  6400] loss: 1.3294165 time : 10.643 s
[epoch 2,imgs 12800] loss: 1.2176111 time : 10.469 s
[epoch 2,imgs 19200] loss: 1.2371912 time : 10.491 s
[epoch 2,imgs 25600] loss: 1.2037246 time : 10.582 s
[epoch 2,imgs 32000] loss: 1.1343563 time : 10.525 s
[epoch 2,imgs 38400] loss: 1.0622729 time : 10.452 s
[epoch 2,imgs 44800] loss: 1.0718397 time : 10.553 s
[epoch 3,imgs  6400] loss: 0.9402382 time : 10.353 s
[epoch 3,imgs 12800] loss: 0.9463086 time : 10.488 s
[epoch 3,imgs 19200] loss: 0.9165696 time : 10.478 s
[epoch 3,imgs 25600] loss: 0.8886583 time : 10.588 s
[epoch 3,imgs 32000] loss: 0.8679459 time : 10.609 s
[epoch 3,imgs 38400] loss: 0.8511683 time : 10.512 s
[epoch 3,imgs 44800] loss: 0.8474520 time : 10.516 s
[epoch 4,imgs  6400] loss: 0.7390380 time : 10.552 s
[epoch 4,imgs 12800] loss: 0.7297320 time : 10.569 s
[epoch 4,imgs 19200] loss: 0.7229294 time : 10.590 s
[epoch 4,imgs 25600] loss: 0.7283565 time : 10.648 s
[epoch 4,imgs 32000] loss: 0.7100309 time : 10.470 s
[epoch 4,imgs 38400] loss: 0.6858836 time : 10.552 s
[epoch 4,imgs 44800] loss: 0.7036798 time : 10.506 s
[epoch 5,imgs  6400] loss: 0.6060488 time : 10.550 s
[epoch 5,imgs 12800] loss: 0.6336668 time : 10.532 s
[epoch 5,imgs 19200] loss: 0.5945961 time : 10.525 s
[epoch 5,imgs 25600] loss: 0.6038743 time : 10.599 s
[epoch 5,imgs 32000] loss: 0.5902489 time : 10.445 s
[epoch 5,imgs 38400] loss: 0.6107914 time : 10.466 s
[epoch 5,imgs 44800] loss: 0.5692267 time : 10.634 s
[epoch 6,imgs  6400] loss: 0.4872262 time : 10.433 s
[epoch 6,imgs 12800] loss: 0.4861911 time : 10.509 s
[epoch 6,imgs 19200] loss: 0.5100237 time : 10.457 s
[epoch 6,imgs 25600] loss: 0.4952321 time : 10.502 s
[epoch 6,imgs 32000] loss: 0.4992105 time : 10.592 s
[epoch 6,imgs 38400] loss: 0.4955591 time : 10.418 s
[epoch 6,imgs 44800] loss: 0.4854409 time : 10.485 s
[epoch 7,imgs  6400] loss: 0.4089725 time : 10.492 s
[epoch 7,imgs 12800] loss: 0.4224316 time : 10.503 s
[epoch 7,imgs 19200] loss: 0.4009730 time : 10.644 s
[epoch 7,imgs 25600] loss: 0.4177441 time : 10.590 s
[epoch 7,imgs 32000] loss: 0.3933671 time : 10.451 s
[epoch 7,imgs 38400] loss: 0.4109715 time : 10.508 s
[epoch 7,imgs 44800] loss: 0.3945034 time : 10.619 s
[epoch 8,imgs  6400] loss: 0.3153646 time : 10.780 s
[epoch 8,imgs 12800] loss: 0.3214465 time : 10.684 s
[epoch 8,imgs 19200] loss: 0.3332480 time : 10.541 s
[epoch 8,imgs 25600] loss: 0.3294883 time : 10.550 s
[epoch 8,imgs 32000] loss: 0.3514154 time : 10.435 s
[epoch 8,imgs 38400] loss: 0.3399832 time : 10.350 s
[epoch 8,imgs 44800] loss: 0.3277325 time : 10.481 s
[epoch 9,imgs  6400] loss: 0.2485025 time : 10.432 s
[epoch 9,imgs 12800] loss: 0.2631959 time : 10.538 s
[epoch 9,imgs 19200] loss: 0.2613422 time : 10.522 s
[epoch 9,imgs 25600] loss: 0.2576190 time : 10.429 s
[epoch 9,imgs 32000] loss: 0.2602800 time : 10.414 s
[epoch 9,imgs 38400] loss: 0.2677160 time : 10.495 s
[epoch 9,imgs 44800] loss: 0.2721200 time : 10.401 s
[epoch 10,imgs  6400] loss: 0.1816192 time : 10.377 s
[epoch 10,imgs 12800] loss: 0.2024573 time : 10.396 s
[epoch 10,imgs 19200] loss: 0.1939250 time : 10.507 s
[epoch 10,imgs 25600] loss: 0.2061785 time : 10.416 s
[epoch 10,imgs 32000] loss: 0.2048160 time : 10.441 s
[epoch 10,imgs 38400] loss: 0.2189552 time : 10.462 s
[epoch 10,imgs 44800] loss: 0.2204565 time : 10.463 s
train end, AlexNet batch_size : 64 train time : 836.921 s

Accuracy of the network on the 10000 test images: 81 %