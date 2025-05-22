# Feature-aware Modulation for Learning from Temporal Tabular Data

This paper is submitted to **NeurIPS'25** (Submission **28660**), under review. 

The experiments share the same setup with Cai & Ye (2025) [1].

## Setup

```bash
conda create --name benchmark python=3.10
pip install -r requirements.txt
conda install faiss-gpu -c pytorch          # only for TabR
```

## Usage Instructions

### Deep method

For deep methods, run:

```bash
python train_model_deep.py --dataset $DATASET_NAME \
                           --model_type $MODEL_NAME \
                           --cat_policy $CAT_POLICY \
                           --enable_timestamp \
                           --gpu 0 --max_epoch 200 --seed_num 15 \
                           --validate_option holdout_foremost_sample \
                           --tune --retune --n_trials 100
```

- `DATASET_NAME`: Dataset name in TabReD benchmark [2].

  ```bash
  choices=(cooking-time, delivery-eta, ecom-offers, homecredit-default,
           homesite-insurance, maps-routing, sberbank-housing, weather)
  ```

- `MODEL_NAME`: Deep method name. `*_temporal` means baseline in [1], `*_modulated` means model with our temporal modulation.

  ```bash
  choices=(
      mlp,        mlp_temporal,        mlp_modulated,
      mlp_plr,    mlp_plr_temporal,    mlp_plr_modulated,
      tabm,       tabm_temporal,       tabm_modulated,
      snn,        snn_temporal,
      dcn2,       dcn2_temporal,
      ftt,        ftt_temporal,
      tabr,       tabr_temporal,
      modernNCA,  modernNCA_temporal,
  )
  ```

- `CAT_POLICY`: Categorical feature policy. We follow [1] and fix this policy to one-hot encoding.

  ```bash
  case $method in
      modernNCA*|tabr*) 
          cat_policy=tabr_ohe
          ;;
      mlp_plr*|tabm*|ftt*|dcn2*|snn*)
          cat_policy=indices
          ;;
      *)
          cat_policy=ohe
          ;;
  esac
  ```

### Classical method

For classical methods, run:

```bash
python train_model_classical.py --dataset $DATASET_NAME \
                                --model_type $MODEL_NAME \
                                --cat_policy $CAT_POLICY \
                                --enable_timestamp \
                                --gpu "" --seed_num 15 \
                                --validate_option holdout_foremost_sample \
                                --tune --retune --n_trials 100
```

- `DATASET_NAME` share the same choices with deep methods.

- `MODEL_NAME`: Classical method name.

  ```bash
  choices=(
      XGBoost, 
      LightGBM, 
      CatBoost, 
      RandomForest, 
      SGD,           # Linear in paper
  )
  ```

- `CAT_POLICY`: Categorical feature policy. We follow [1] and fix this policy to one-hot encoding.

  ```bash
  case $method in
      catboost)
          cat_policy=indices
          ;;
      *)
          cat_policy=ohe
          ;;
  esac
  ```

**Enjoy the code!** 

---

[1] Cai, H.-R. and Ye, H.-J. Understanding the limits of deep tabular methods with temporal shift. In ICML, 2025.

[2] Rubachev, I., Kartashev, N., Gorishniy, Y., and Babenko, A. Tabred: A benchmark of tabular machine learning in-the-wild. In ICLR, 2025.