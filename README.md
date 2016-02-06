# InsultDetector
Данный скрипт решает задачу классификации текстов по наличию/отсутствию в них личных оскорблений. Задача решена с помощью машинного обучения и библиотеки на python3 scikit-learn.
Был использован классификатор PassiveAgressiveClissifier, который показал лучшие результаты и при данном извлечении признаков (HashingVectorizer(), TF-IDF). Также, был применен метод латентного семантического анализа (в библиотеке scikit-learn реализован в виде TruncatedSVD()), который улучшил результат на ~10% (по F1-мере), затрачивая при этом на несколько порядков больше времени. Поэтому применение LSA(латентно-семантического анализа) при ограничении по времени не оправдано. 
<br>
<br>
This script detects personal insult in texts. It was used python3 scikit-learn for implementig, in particular PassiveAgressiveClassifier. The choice justified by F1-score, that shows best score of PassiveAgressiveClassifier. 
LSA (latent semantic analysis) shows better score, but takes ~exp increasing time. So that, TruncatedSVD(=LSA in scikit-learn) is not useful in realtime.
