#include "simple_svm.h"

/* 
 * 処理マクロ
 */

/* 対称性a(i,j) = a(j,i)を持つデータ構造内で, num_elemだけある要素の中で動くインデックスi,jからアクセスするインデックスを決定する */
#define SYMMETRIC_INDEX_AT(num_elem, i, j) \
  (((i) <= (j)) ? ((i)*(num_elem)-(i)*((i)-1)/2+((j)-(i))) : ((j)*(num_elem)-(j)*((j)-1)/2 + ((i)-(j))))

/* グラム行列へのアクセスマクロ */
#define GRAM_MATRIX_AT(gram_mat, num_column, i, j) \
  (gram_mat)[SYMMETRIC_INDEX_AT((num_column), (i), (j))]

/* NULLチェックを行い、freeをする */
#define NULLCHECK_AND_FREE(p) \
{ \
  if ((p) != NULL) { free(p); } \
} 

/* 
 * スタティック関数宣言
 */

/* カーネル関数 */
static SmpSVMReal linear_kernel(const SmpSVMReal *input_x, const SmpSVMReal *input_y, const size_t vector_size, const SmpSVMReal *kernel_parameter); /* 線形カーネル */
static SmpSVMReal gaussian_kernel(const SmpSVMReal *input_x, const SmpSVMReal *input_y, const size_t vector_size, const SmpSVMReal *kernel_parameter); /* ガウシアンカーネル */
static SmpSVMReal polynormal_kernel(const SmpSVMReal *input_x, const SmpSVMReal *input_y, const size_t vector_size, const SmpSVMReal *kernel_parameter); /* 多項式カーネル */

/* サンプルのクリア */
static void smpsvm_clear_sample(SmpSVMHn handle);
/* グラム行列を再計算 */
static void smpsvm_recalc_gram_matrix(SmpSVMHn handle);
/* データの正規化 */
static void smpsvm_normalize_data(const SmpSVMHn handle, SmpSVMReal *data, const size_t data_dim);

SmpSVMHn smpSVM_Create(const SmpSVMConfig *config)
{
  SmpSVMHn handle;

  if (config == NULL) {
    fprintf(stderr, "Invalid parameter: pointer to config point to NULL. \n");
    return NULL;
  }

  /* ハンドルの領域割当て */
  handle = (SmpSVMHn)malloc(sizeof(struct SmpSVMTag));
  if (handle == NULL) {
    fprintf(stderr, "Failed to craete SVM handler. \n");
    return NULL;
  }

  /* コンフィグ値をチェック */
  if (config->learning_rate <= 0
      || config->soft_margin_const < 0) {
    fprintf(stderr, "Invalid config parameter. \n");
    return NULL;
  }

  /* コンフィグをセット */
  handle->learning_rate    = config->learning_rate;
  handle->soft_margin_type = config->soft_margin_type;
  handle->soft_margin_const = config->soft_margin_const;

  /* ポインタをクリア */
  handle->sample_data      = NULL;
  handle->sample_label     = NULL;
  handle->gram_matrix      = NULL;
  handle->sample_maxmin[0] = NULL;
  handle->sample_maxmin[1] = NULL;
  handle->dual_coef        = NULL;

  /* デフォルトのカーネル関数をセット */
  handle->kernel_function  = gaussian_kernel;
  handle->kernel_parameter = (SmpSVMReal *)malloc(sizeof(SmpSVMReal));
  handle->kernel_parameter[0] = 0.5f;

  /* SVMの状態をサンプル未セットに更新 */
  handle->status           = SMPSVM_STATUS_NOT_SET_SAMPLE;

  return handle;
}

void smpSVM_Destroy(SmpSVMHn handle)
{
  size_t i_sample;
  if (handle != NULL) {
    NULLCHECK_AND_FREE(handle->sample_label);
    NULLCHECK_AND_FREE(handle->gram_matrix);
    NULLCHECK_AND_FREE(handle->sample_maxmin[0]);
    NULLCHECK_AND_FREE(handle->sample_maxmin[1]);
    NULLCHECK_AND_FREE(handle->dual_coef);
    /* サンプルデータの解放 */
    if (handle->sample_data != NULL) {
      for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
        NULLCHECK_AND_FREE(handle->sample_data[i_sample]);
      }
    }
    free(handle);
  }
}

int smpSVM_SetSample(SmpSVMHn handle, const SmpSVMReal **sample_data, const int *sample_label, const size_t sample_num, const size_t sample_dim)
{
  size_t i_dim, i_sample, i_x, i_y;
  SmpSVMReal data_buf;
  SmpSVMReal *normalized_data;

  if (handle == NULL) {
    fprintf(stderr, "SVM handler points to NULL. \n");
    return -1;
  }

  /* サンプルを一旦クリア */
  smpsvm_clear_sample(handle);

  /* サンプル個数と次元を取得 */
  handle->sample_num       = sample_num;
  handle->sample_dimension = sample_dim;

  /* サンプルデータの割当と取得 */
  handle->sample_data = (SmpSVMReal **)malloc(sizeof(SmpSVMReal*) * sample_num);
  handle->sample_maxmin[0] = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * sample_dim);
  handle->sample_maxmin[1] = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * sample_dim);
  if (handle->sample_data == NULL
      || handle->sample_maxmin[0] == NULL
      || handle->sample_maxmin[1] == NULL) {
    fprintf(stderr, "Failed to allocate sample data. \n");
    return -1;
  }

  /* 最大値と最小値の初期化 */
  for (i_dim = 0; i_dim < sample_dim; ++i_dim) {
    handle->sample_maxmin[0][i_dim] = -FLT_MAX;
    handle->sample_maxmin[1][i_dim] = FLT_MAX;
  }

  for (i_sample = 0; i_sample < sample_num; ++i_sample) {
    handle->sample_data[i_sample] = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * sample_dim);
    if (handle->sample_data[i_sample] == NULL) {
      fprintf(stderr, "Failed to allocate sample data. \n");
      return -2;
    }
    memcpy(handle->sample_data[i_sample], 
           sample_data[i_sample], 
           sizeof(SmpSVMReal) * sample_dim);

    /* 最大値と最小値の更新 */
    for (i_dim = 0; i_dim < sample_dim; ++i_dim) {
      data_buf = sample_data[i_sample][i_dim];
      if (handle->sample_maxmin[0][i_dim] < data_buf) {
        handle->sample_maxmin[0][i_dim] = data_buf;
      } 
      if (handle->sample_maxmin[1][i_dim] > data_buf) {
        handle->sample_maxmin[1][i_dim] = data_buf;
      }
    }
  }
  
  /* サンプルデータの正規化 */
  normalized_data = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * sample_dim);
  if (handle->sample_data[i_sample] == NULL) {
    fprintf(stderr, "Failed to allocate working buffer. \n");
    return -3;
  }
  for (i_sample = 0; i_sample < sample_num; ++i_sample) {
    memcpy(normalized_data, handle->sample_data[i_sample], sizeof(SmpSVMReal) * sample_dim);
    smpsvm_normalize_data(handle, normalized_data, sample_dim);
    memcpy(handle->sample_data[i_sample], normalized_data, sizeof(SmpSVMReal) * sample_dim);
  }
  
  /* サンプルラベルの取得 */
  handle->sample_label = (int *)malloc(sizeof(int) * sample_num);
  if (handle->sample_label == NULL) {
    fprintf(stderr, "Failed to allocate sample label. \n");
    return -4;
  }
  memcpy(handle->sample_label, sample_label, sizeof(int) * sample_num);

  /* グラム行列の割当て/初期化 */
  /* グラム行列は対称なので, サイズnに対してn*(n+1)/2で十分 */
  handle->gram_matrix = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * sample_num * (sample_num+1) / 2);
  if (handle->gram_matrix == NULL) {
    fprintf(stderr, "Failed to allocate gram matrix. \n");
    return -5;
  }
  for (i_x = 0; i_x < sample_num; ++i_x) {
    for (i_y = i_x; i_y < sample_num; ++i_y) {
      GRAM_MATRIX_AT(handle->gram_matrix, sample_num, i_x, i_y) = 0.0f;
    }
  }
  /* グラム行列の計算 */
  smpsvm_recalc_gram_matrix(handle);

  /* ハンドルの状態を更新 */
  handle->status = SMPSVM_STATUS_NOT_LEARNED;

  /* 作業領域の解放 */
  free(normalized_data);
  return 0;
}

int smpSVM_SetConfig(SmpSVMHn handle, const SmpSVMConfig *config)
{
  if (handle == NULL || config == NULL) {
    fprintf(stderr, "Invalid parameter: handle or config points to NULL. \n");
    return -1;
  }

  /* コンフィグを適用 */
  handle->learning_rate    = config->learning_rate;
  handle->soft_margin_type = config->soft_margin_type;
  handle->soft_margin_const = config->soft_margin_const;

  return 0;
}

int smpSVM_SetKernelFunction(SmpSVMHn handle, const kernel_function_t kernel_function, const SmpSVMReal *kernel_parameter, const size_t kernel_parameter_num)
{
  if (kernel_function == NULL || kernel_parameter == NULL) {
    fprintf(stderr, "Invalid parameter: kernel_function or kernel_parameter points NULL. \n");
    return -1;
  }

  handle->kernel_parameter = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * kernel_parameter_num);
  if (handle->kernel_parameter == NULL) {
    fprintf(stderr, "Failed to allocate kernel parameter. \n");
    return -2;
  }

  /* カーネル関数とパラメタをセット */
  handle->kernel_function = kernel_function;
  memcpy(handle->kernel_parameter, kernel_parameter, sizeof(SmpSVMReal) * kernel_parameter_num);

  return 0;
}

int smpSVM_SetDualCoefficients(SmpSVMHn handle, const SmpSVMReal *dual_coef, const size_t sample_num)
{
  if (handle == NULL || dual_coef == NULL) {
    fprintf(stderr, "Invalid parameter: handle or dual_coef points to NULL. \n");
    return -1;
  }

  if (handle->sample_num != sample_num) {
    fprintf(stderr, "Invalid parameter: number of sample differ from handle (handle:%zd, arg:%zd). \n", handle->sample_num, sample_num);
    return -2;
  }

  handle->dual_coef = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * sample_num);
  if (handle->dual_coef == NULL) {
    fprintf(stderr, "Failed to allocate dual coefficients.\n ");
    return -3;
  }

  /* 双対係数の取得 */
  memcpy(handle->dual_coef, dual_coef, sizeof(SmpSVMReal) * sample_num);
  /* SVMの状態を更新 */
  handle->status = SMPSVM_STATUS_SET_COEF;
  return 0;

}

/* さぁ...学習の時間だ */
int smpSVM_Learning(SmpSVMHn handle)
{
  size_t iteration;
  size_t i_sample, i_x, i_y;
  SmpSVMReal soft_margin_C1, soft_margin_C2;         /* ソフトマージン定数の係数値 */
  SmpSVMReal *diff_dual_coef, *pre_diff_dual_coef;   /* 双対係数の勾配, 双対係数の前ステップの勾配 */
  SmpSVMReal *pre_dual_coef;                         /* 双対係数の前ステップの値 */
  SmpSVMReal diff_sum, coef_dist, kernel_val;        /* 勾配の和, 係数の変化距離, カーネル関数値 */
  SmpSVMReal coef_diff, diff_dist, dual_coef_average;    /* 係数の変位, 勾配の変化距離, ラベル付き双対係数和 */
  /* TODO:diffとかdistとか名前がわかりづらい！ */

  if (handle == NULL) {
    fprintf(stderr, "Invalid parameter: handle points to NULL. \n");
    return -1;
  }

  /* ハンドルの状態を検査: サンプルをセットしていなければだめ */
  if (handle->status == SMPSVM_STATUS_NOT_SET_SAMPLE) {
    fprintf(stderr, "SVM handle is not set sample data. \n");
    return -2;
  }

  /* 作業領域を割り当て */
  diff_dual_coef     = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * handle->sample_num);
  pre_diff_dual_coef = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * handle->sample_num);
  pre_dual_coef      = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * handle->sample_num);
  if (diff_dual_coef == NULL 
      || pre_diff_dual_coef == NULL
      || pre_dual_coef == NULL) {
    fprintf(stderr, "Failed to allocate working buffer. \n");
    return -3;
  }

  /* ソフトマージンの定数をセット */
  switch (handle->soft_margin_type) {
    case SMPSVM_HARDMARGIN: 
      soft_margin_C1 = (SmpSVMReal)FLT_MAX;
      soft_margin_C2 = (SmpSVMReal)FLT_MAX;
      break;
    case SMPSVM_SOFTMARGIN_ONE_NORM:
      soft_margin_C1 = handle->soft_margin_const;
      soft_margin_C2 = (SmpSVMReal)FLT_MAX;
      break;
    case SMPSVM_SOFTMARGIN_TWO_NORM:
      soft_margin_C1 = (SmpSVMReal)FLT_MAX;
      soft_margin_C2 = handle->soft_margin_const;
      break;
    default:
      fprintf(stderr, "Not supported soft margin type. \n");
      return -4;
  }

  /* 双対係数の初期化 */
  handle->dual_coef = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * handle->sample_num);
  if (handle->dual_coef == NULL) {
    fprintf(stderr, "Failed to allocate dual coefficients.\n ");
    return -5;
  }

  /* 乱数によって初期化 */
  for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
    if (handle->sample_label[i_sample] == 0) {
      /* 欠損データの係数は0にして使用しない */
      handle->dual_coef[i_sample] = 0.0f;
    } else {
      handle->dual_coef[i_sample] = (SmpSVMReal)rand() / RAND_MAX;
    }
    /* 勾配は0で初期化 */
    diff_dual_coef[i_sample]
      = pre_diff_dual_coef[i_sample] = 0.0f;
  }

  /* グラム行列を再計算 */
  smpsvm_recalc_gram_matrix(handle);
  
  /* 状態の初期化 */
  iteration = 0;
  diff_sum = coef_dist = (SmpSVMReal)FLT_MAX;
  handle->status = SMPSVM_STATUS_NOT_LEARNED;

  /* 学習ループ */
  while ( iteration < SMPSVM_MAX_LEARN_ITERATION ) {
    /* 前回の係数, 勾配を保存 */
    memcpy(pre_dual_coef, handle->dual_coef, sizeof(SmpSVMReal) * handle->sample_num);
    memcpy(pre_diff_dual_coef, diff_dual_coef, sizeof(SmpSVMReal) * handle->sample_num);

    /* 勾配値の計算 */
    diff_dist = 0.0f;
    for (i_x = 0; i_x < handle->sample_num; ++i_x) {
      diff_sum = 0.0f;
      for (i_y = 0; i_y < handle->sample_num; ++i_y) {
        /* C2を踏まえたカーネル関数値を計算 */
        kernel_val = GRAM_MATRIX_AT(handle->gram_matrix, handle->sample_num, i_x, i_y);
        if (i_x == i_y) {
          kernel_val += (1.0f / soft_margin_C2);
        }
        diff_sum += handle->dual_coef[i_y] * handle->sample_label[i_y] * kernel_val;
      }
      diff_sum *= handle->sample_label[i_x];
      diff_dual_coef[i_x] = 1.0f - diff_sum;
      diff_dist += (1.0f - diff_sum) * (1.0f - diff_sum);
    }
    
    /* 双対係数の更新 */
    for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
      if ( handle->sample_label[i_sample] == 0 ) {
        continue;
      }
      /* printf("dual_coef[%d]:%f -> ", i_sample, handle->dual_coef[i_sample]); */
      handle->dual_coef[i_sample] 
        += handle->learning_rate * (diff_dual_coef[i_sample] + SMPSVM_MOMENT_RATE * pre_diff_dual_coef[i_sample]); 
      /* printf("%f \n", handle->dual_coef[i_sample]); */

      /* 非数,無限チェック */
      if ( isnan(handle->dual_coef[i_sample]) || isinf(handle->dual_coef[i_sample]) ) {
        fprintf(stderr, "Detected NaN or Inf Dual-Coffience. \n");
        return -3;
      }
    }

    /* 双対係数へ制約を適用 */
    /* 制約1: 正例と負例の双対係数和を等しくする. */
    dual_coef_average = 0.0f;
    for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
      dual_coef_average 
        += (handle->sample_label[i_sample] * handle->dual_coef[i_sample]);
    }
    dual_coef_average /= handle->sample_num;
    for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
      if ( handle->sample_label[i_sample] == 0 ) {
        continue;
      }
      handle->dual_coef[i_sample] 
        -= (dual_coef_average / handle->sample_label[i_sample]);
    }

    /* 制約2: 双対係数は非負 */
    coef_dist = 0.0f;
    for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
      if ( handle->dual_coef[i_sample] < 0.0f ) {
        handle->dual_coef[i_sample] = 0.0f;
      } else if ( handle->dual_coef[i_sample] > soft_margin_C1 ) {
        /* C1ノルムの制約を適用 */
        handle->dual_coef[i_sample] = soft_margin_C1;
      }
      /* ここで最終結果が出る. 前回との変化を計算 */
      coef_diff = pre_dual_coef[i_sample] - handle->dual_coef[i_sample];
      coef_dist += (coef_diff * coef_diff);
    }

    /* 収束判定 */
    if (sqrt(coef_dist) < SMPSVM_REAL_VALUE_EPSILON
        || sqrt(diff_dist) < SMPSVM_REAL_VALUE_EPSILON) {
      handle->status = SMPSVM_STATUS_LEARNED;
      break;
    }

    /* 学習繰り返し回数の増加 */
    iteration++;
    /* 結果を印字 */
    printf("ite: %zd diff_dist:%f coef_dist:%f \n", iteration, sqrt(diff_dist), sqrt(coef_dist));
  }

  if (iteration >= SMPSVM_MAX_LEARN_ITERATION) {
    fprintf(stderr, "Warning: Learning process not convergenced. \n");
    handle->status = SMPSVM_STATUS_LEARNED;
    // return -5;
  } 

  /* 作業領域を解放 */
  free(diff_dual_coef);
  free(pre_diff_dual_coef);

  return 0;

}

int smpSVM_PredictLabel(const SmpSVMHn handle, int *result, const SmpSVMReal *data, const size_t data_dim)
{
  size_t     i_sample;
  SmpSVMReal network_output;   /* SVMの双対ネットワークの出力値 */
  SmpSVMReal *normalized_data; /* 正規化したデータ */


  /* 引数チェック */
  if (handle == NULL || result == NULL || data == NULL) {
    fprintf(stderr, "Invalid parameter: parameter(handle or label result or data) points to NULL. \n");
    return -1;
  }
  if (handle->sample_dimension != data_dim) {
    fprintf(stderr, "Invalid parameter: data dimension conflicted(handle:%zd, input:%zd). \n", handle->sample_dimension, data_dim);
    return -2;
  }

  /* 学習が終了していない, 若しくは係数がセットされていない
   * 場合は、まだ予測できない */
  if (handle->status != SMPSVM_STATUS_LEARNED
      && !(handle->status == SMPSVM_STATUS_SET_COEF && handle->sample_data == NULL)) {
    fprintf(stderr, "SVM handler is not learned yet. \n");
    return -3;
  }

  normalized_data = (SmpSVMReal *)malloc(sizeof(SmpSVMReal) * data_dim);
  if (normalized_data == NULL) {
    fprintf(stderr, "Failed to allocate working buffer.\n");
    return -4;
  }

  /* 正規化 */
  memcpy(normalized_data, data, sizeof(SmpSVMReal) * data_dim);
  smpsvm_normalize_data(handle, normalized_data, data_dim);

  /* ネットワーク出力計算 */
  network_output = 0.0f;
  for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
    /* 係数が正に相当するサンプル（サポートベクトル）
     * のみを計算する */
    if (handle->dual_coef[i_sample] > 0.0f) {
      network_output 
        += handle->sample_label[i_sample] * handle->dual_coef[i_sample]
        * handle->kernel_function(handle->sample_data[i_sample], normalized_data, data_dim, handle->kernel_parameter);
    }
  }

  /* 識別 */
  *result = (network_output >= 0.0f) ? 1 : -1;

  return 0;

}

static void smpsvm_clear_sample(SmpSVMHn handle)
{
  size_t i_sample;

  /* サンプルデータの解放 */
  if (handle->sample_data != NULL) {
    for (i_sample = 0; i_sample < handle->sample_num; ++i_sample) {
      NULLCHECK_AND_FREE(handle->sample_data[i_sample]);
    }
  }

  /* サンプルに関わるデータを解放 */
  NULLCHECK_AND_FREE(handle->sample_label);
  NULLCHECK_AND_FREE(handle->gram_matrix);
  NULLCHECK_AND_FREE(handle->sample_maxmin[0]);
  NULLCHECK_AND_FREE(handle->sample_maxmin[1]);

  /* サンプルに関わる情報をクリア */
  handle->sample_dimension = 0;
  handle->sample_num       = 0;

  /* SVMの状態を更新 */
  handle->status = SMPSVM_STATUS_NOT_SET_SAMPLE;

}

static void smpsvm_recalc_gram_matrix(SmpSVMHn handle)
{
  size_t i_x, i_y;
  for (i_x = 0; i_x < handle->sample_num; ++i_x) {
    for (i_y = i_x; i_y < handle->sample_num; ++i_y) {
      GRAM_MATRIX_AT(handle->gram_matrix, handle->sample_num, i_x, i_y)
        = handle->kernel_function(
            handle->sample_data[i_x],
            handle->sample_data[i_y],
            handle->sample_dimension,
            handle->kernel_parameter);
    }
  }
}

static void smpsvm_normalize_data(const SmpSVMHn handle, SmpSVMReal *data, const size_t data_dim)
{
  size_t i_dim;
  for (i_dim = 0; i_dim < data_dim; ++i_dim) {
    data[i_dim]
      = ( data[i_dim] - handle->sample_maxmin[1][i_dim] ) / ( handle->sample_maxmin[0][i_dim] - handle->sample_maxmin[1][i_dim] );
  }
}

/* 線形カーネル */
static SmpSVMReal linear_kernel(const SmpSVMReal *input_x, const SmpSVMReal *input_y, const size_t vector_size, const SmpSVMReal *kernel_parameter)
{
  size_t i_dim;
  SmpSVMReal ret = 0.0f, norm_x = 0.0f, norm_y = 0.0f;

  /* 標準内積 */
  for (i_dim = 0; i_dim < vector_size; ++i_dim) {
    ret    += input_x[i_dim] * input_y[i_dim];
    norm_x += input_x[i_dim] * input_x[i_dim];
    norm_y += input_y[i_dim] * input_y[i_dim];
  }

  if (norm_x == 0.0f || norm_y == 0.0f) {
    return 0.0f;
  }

  return ret / sqrt(norm_x * norm_y);
}

/* ガウシアンカーネル */
static SmpSVMReal gaussian_kernel(const SmpSVMReal *input_x, const SmpSVMReal *input_y, const size_t vector_size, const SmpSVMReal *kernel_parameter)
{
  size_t i_dim;
  SmpSVMReal dist;

  /* ベクトル間のユーグリッド距離を計算 */
  dist = 0.0f;
  for (i_dim = 0; i_dim < vector_size; ++i_dim) {
    dist += powf(input_x[i_dim] - input_y[i_dim], 2);
  }

  /* note:この式には色んな形式があることに留意せよ */
  return exp( -dist / powf(kernel_parameter[0],2) );
}

/* 多項式カーネル */
static SmpSVMReal polynormal_kernel(const SmpSVMReal *input_x, const SmpSVMReal *input_y, const size_t vector_size, const SmpSVMReal *kernel_parameter) 
{
  size_t i_dim;
  SmpSVMReal innprod_val;

  /* 標準内積を計算 */
  innprod_val = 0.0f;
  for (i_dim = 0; i_dim < vector_size; ++i_dim) {
    innprod_val += input_x[i_dim] * input_y[i_dim];
  }

  return powf( (innprod_val + kernel_parameter[0]), kernel_parameter[1]);
}
