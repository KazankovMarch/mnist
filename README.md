# mnist
Нейронная сеть, обучаемая распознавать цифры.

Для тестирования уже обученной нейросети нужно запустить скрипт `run_model.py`.
Если нужно протестировать на собственных изображениях - нужно добавить изображения в папку `images` и запустить скрипт с ключом `--uci`. (изображения должны быть 28x28 пикселей с черным фоном и одной белой цифрой, расширение - png)
Если нужно простроить матрицу ошибок, нужно добавить ключ `--bcm`.

Для пересоздания нейросети нужно запустить скрипт `create.py`.
