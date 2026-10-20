# Эксперименты: сравнение uplift-подходов кросс-валидацией

Скрипт `exp_uplift_cv.py` сравнивает семь подходов на 15 разбиениях (5 блоков × 3 повтора со стратификацией по `treatment` и `target`) и считает Uplift@30, Uplift AUC и Qini AUC. Перед сравнением он воспроизводит результаты основного ноутбука (разбиение 80/20, `random_state=42`), чтобы убедиться, что окружение даёт те же значения.

Результаты сохранены в `results/uplift_cv.json`; таблица и график разбросов показаны в ноутбуке в разделе «Устойчивость выбора модели».

## Результаты (15 разбиений)

| Модель | Uplift@30, среднее | std | Uplift AUC | Qini AUC |
|---|---:|---:|---:|---:|
| UpliftRF (как в ноутбуке) | 0,0415 | 0,0088 | 0,0238 | 0,0499 |
| UpliftRF, критерий KL, глубина 7 | 0,0406 | 0,0099 | 0,0224 | 0,0471 |
| T-learner RF | 0,0402 | 0,0083 | 0,0218 | 0,0452 |
| UpliftRF, критерий ED, глубина 6 | 0,0386 | 0,0087 | 0,0245 | 0,0514 |
| S-learner RF | 0,0352 | 0,0105 | 0,0212 | 0,0439 |
| X-learner RF (propensity score) | 0,0340 | 0,0103 | 0,0213 | 0,0442 |
| S-learner CatBoost | 0,0340 | 0,0091 | 0,0242 | 0,0504 |

## Запуск в Docker

Образ собирается из `requirements.txt` проекта (из корня репозитория):

```bash
docker build -f experiments/Dockerfile -t uplift-project .
```

Библиотека курса клонируется в папку `scikit_uplift` (нужна только основному ноутбуку, скрипту эксперимента не нужна). Запуск эксперимента (около часа на 8 ядрах):

```bash
docker run --rm -v "$(pwd)":/app uplift-project python experiments/exp_uplift_cv.py
```

Быстрая проверка (два блока, одна модель, без воспроизведения):

```bash
docker run --rm -v "$(pwd)":/app uplift-project \
  python experiments/exp_uplift_cv.py --splits 2 --repeats 1 --only "S-learner RF" --skip-repro --out /tmp/check.json
```

В Windows PowerShell вместо `$(pwd)` используйте `${PWD}` или полный путь.
