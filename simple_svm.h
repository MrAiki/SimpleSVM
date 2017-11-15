#ifndef _SIMPLE_SVM_H_INCLUDED_
#define _SIMPLE_SVM_H_INCLUDED_

/*!
 * @file simple_svm.h
 * @brief Simple SVM implementation in C language
 *        C言語による単純なSVMの実装
 */

/* include headers インクルード */
/* C standard headers C標準ライブラリ */
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 
 * Constant macro definition 定数マクロ定義
 */
#define SMPSVM_MAX_LEARN_ITERATION       (5000)     /* max learning iteration count 学習最大繰り返し回数 */
#define SMPSVM_REAL_VALUE_EPSILON        (0.0001f)  /* small real-value for detect convergence 収束判定用の小さな値 */
#define SMPSVM_MOMENT_RATE               (0.2f)     /* ratio for learning rate to moment rate 慣性項の学習率に対する比率 */
#define SMPSVM_DEFAULT_LEARNING_RATE     (0.05f)    /* default learning rate 学習率のデフォルト値 */
#define SMPSVM_DEFAULT_SOFT_MARGIN_TYPE  (0)        /* default soft-margin type デフォルトのソフトマージンの種類 */
#define SMPSVM_DEFAULT_SOFT_MARGIN_CONST (FLT_MAX)  /* default soft-margin constant ソフトマージンのデフォルト値 */

/*
 * Processing macro definition 処理マクロ定義
 */
/* Set default config デフォルトのコンフィグ値をセット */
#define smpSVM_SetDefaultConfig(config_p) \
{ \
  SmpSVMConfig *tmp_p     = config_p; \
  tmp_p->learning_rate    = SMPSVM_DEFAULT_LEARNING_RATE; \
  tmp_p->soft_margin_type = SMPSVM_DEFAULT_SOFT_MARGIN_TYPE; \
  tmp_p->soft_margin_const = SMPSVM_DEFAULT_SOFT_MARGIN_CONST; \
} \

/*
 * type definition 型定義
 */
/* real value 実数値 */
typedef float SmpSVMReal;
/* note: If you want double-precision calculation result, please set float to double. 倍精度で演算したい場合はfloatをdoubleにしてね */

/* @brief kernel function カーネル関数
 * @param[in] input_x kernel function input vector カーネル関数への入力ベクトル
 * @param[in] input_y kernel function input vector カーネル関数への入力ベクトル
 * @param[in] vector_size input vector size 入力ベクトルのサイズ
 * @param[in] kernel_parameter kernel parameter カーネルパラメタ配列
 * @return Real-valued kernel function output カーネル関数の計算結果（実数値）
 */
typedef SmpSVMReal (*kernel_function_t)(const SmpSVMReal *input_x, const SmpSVMReal *input_y, const size_t vector_size, const SmpSVMReal *kernel_parameter);
/* note: Self-definition kernel function OK. Please use smpSVM_SetKernelFunction. アプリケーション側でカーネル関数を定める事ができます. smpSVM_SetKernelFunctionを使って下さい. */

/* SVM status SVMの状態 */
typedef enum {
  SMPSVM_STATUS_NOT_SET_SAMPLE, /* No sample        サンプルがセットされていない */
  SMPSVM_STATUS_NOT_LEARNED,    /* Not learning yet 学習していない */
  SMPSVM_STATUS_LEARNED,        /* Leaned           学習が正常終了した */
  SMPSVM_STATUS_SET_COEF,       /* Coeffients set   係数がセットされた */
} SmpSVMStatus;

/* Soft-margin type ソフトマージンの種類 */
typedef enum {
  SMPSVM_HARDMARGIN = SMPSVM_DEFAULT_SOFT_MARGIN_TYPE,  /* Hard-margin (do not use soft-margin, default) ハードマージン（ソフトマージンを使わない）:デフォルト */
  SMPSVM_SOFTMARGIN_ONE_NORM,                           /* One-norm soft margin 1-ノルムソフトマージン */
  SMPSVM_SOFTMARGIN_TWO_NORM,                           /* Two-norm soft margin 2-ノルムソフトマージン */
} SmpSVMSoftMarginType;

/* SVM handler SVMのハンドラ */
typedef struct SmpSVMTag {
  SmpSVMReal           **sample_data;        /* Sample data vectors サンプルデータのベクトル */
  int                  *sample_label;        /* Sample data label(-1 or 1 label) サンプルデータのラベル */
  size_t               sample_dimension;     /* Sample data vector dimension サンプルデータベクトルの次元 */
  size_t               sample_num;           /* Number of samples サンプルデータの個数 */
  SmpSVMReal           *gram_matrix;         /* Gram matrix(represented by 1-dimension array) グラム行列（1次元配列で表現） */
  SmpSVMReal           *sample_maxmin[2];    /* Sample max(index:0) and min(index:1) value for normalization 正規化の為のサンプル最大値（第一要素）と最小値（第二要素） */
  SmpSVMReal           *dual_coef;           /* Dual coefficients 双対係数 */
  SmpSVMReal           learning_rate;        /* Learning rate 学習率 */
  SmpSVMSoftMarginType soft_margin_type;     /* Soft-margin type ソフトマージンの種類 */
  SmpSVMReal           soft_margin_const;     /* Soft-margin constant ソフトマージン定数 */
  SmpSVMStatus         status;               /* SVM status SVMの学習状態 */
  kernel_function_t    kernel_function;      /* Kernel function カーネル関数 */
  SmpSVMReal           *kernel_parameter;    /* Kernel parameter カーネル関数パラメタ */
} *SmpSVMHn;

/* SVM handler config SVMハンドラのコンフィグ */
typedef struct SmpSVMConfigTag {
  SmpSVMReal           learning_rate;        /* Learning rate 学習率 */
  SmpSVMSoftMarginType soft_margin_type;     /* Soft-margin type ソフトマージンの種類 */
  SmpSVMReal           soft_margin_const;     /* Soft-margin constant ソフトマージン定数 */
} SmpSVMConfig;

/*
 * Public function declaration 公開関数宣言
 */
/* 
 * @brief Create SVM handler SVMのハンドルを作成
 * @param[in] config SVM Config SVMのコンフィグ 
 * @return SVM handler SVMのハンドラ
 */
SmpSVMHn smpSVM_Create(const SmpSVMConfig *config);

/* 
 * @brief Destroy SVM handle SVMハンドルを破棄
 * @param[in,out] handle   SVM handler       SVMハンドル
 */
void smpSVM_Destroy(SmpSVMHn handle);

/* 
 * @brief Set sample data サンプルデータをセット
 * @param[in,out] handle       SVM handler SVMハンドル
 * @param[in]     sample_data  Sample data サンプルデータ
 * @param[in]     sample_label Sample label データに対応するラベル
 * @param[in]     sample_num   Number of samples サンプルの数
 * @param[in]     sample_dim   Dimension of samples サンプルの数
 * @return Return 0 if function succeeded, otherwise returns negative value. 成功した場合は0を、失敗した場合は負値を返します
 */
int smpSVM_SetSample(SmpSVMHn handle, const SmpSVMReal **sample_data, const int *sample_label, const size_t sample_num, const size_t sample_dim);

/* 
 * @brief Set config コンフィグをセット
 * @param[in,out] handle        SVM handler SVMハンドル
 * @param[in] config SVM Config SVMのコンフィグ 
 * @return Return 0 if function succeeded, otherwise returns negative value. 成功した場合は0を、失敗した場合は負値を返します
 */
int smpSVM_SetConfig(SmpSVMHn handle, const SmpSVMConfig *config);

/* TODO:kernel_functionの戻り値検査 */
/* 
 * @brief Set config コンフィグをセット
 * @param[in,out] handle            SVM handler     SVMハンドル
 * @param[in]     kernel_function   kernel function カーネル関数
 * @param[in]     kernel_parameter  kernel parameter カーネル関数パラメタ
 * @param[in]     kernel_parameter_num   Number of kernel function カーネルパラメタの個数
 * @return Return 0 if function succeeded, otherwise returns negative value. 成功した場合は0を、失敗した場合は負値を返します
 */
int smpSVM_SetKernelFunction(SmpSVMHn handle, const kernel_function_t kernel_function, const SmpSVMReal *kernel_parameter, const size_t kernel_parameter_num);

/* 
 * @brief Set dual coefficients 双対係数をセット
 * @param[in,out] handle        SVM handler       SVMハンドル
 * @param[in]     dual_coef     Dual coefficients 双対係数
 * @param[in]     sample_num    Number of samples サンプルの数
 * @return Return 0 if function succeeded, otherwise returns negative value. 成功した場合は0を、失敗した場合は負値を返します
 */
int smpSVM_SetDualCoefficients(SmpSVMHn handle, const SmpSVMReal *dual_coef, const size_t sample_num);

/* 
 * @brief Learning SVM SVMを学習
 * @param[in,out] handle        SVM handler       SVMハンドル
 * @return Return 0 if function succeeded, otherwise returns negative value. 成功した場合は0を、失敗した場合は負値を返します
 */
int smpSVM_Learning(SmpSVMHn handle);

/* 
 * @brief Predict data label データのラベルを予測
 * @param[in]     handle   SVM handler       SVMハンドル
 * @param[in,out] result   Predicted label   予測ラベル
 * @param[in]     data     Data for predict  予測対象のデータ
 * @param[in]     data_dim Dimension of data 予測対象のデータの次元
 * @return Return 0 if function succeeded, otherwise returns negative value. 成功した場合は0を、失敗した場合は負値を返します
 */
int smpSVM_PredictLabel(const SmpSVMHn handle, int *result, const SmpSVMReal *data, const size_t data_dim);

#endif /* _SIMPLE_SVM_H_INCLUDED_ */

